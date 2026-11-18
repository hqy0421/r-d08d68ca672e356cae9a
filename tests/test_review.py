"""Tests for the versioned human-review training and publishing workflow."""

import json
import os

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import recordlinkage as rl
from recordlinkage.annotation import read_annotation_file
from recordlinkage.annotation import write_annotation_file
from recordlinkage.review import GenerationError
from recordlinkage.review import LabelConflictError
from recordlinkage.review import ReviewLabel
from recordlinkage.review import ReviewPipeline
from recordlinkage.review import ReviewStore
from recordlinkage.review import SchemaMismatchError
from recordlinkage.review import SourceRegistry
from recordlinkage.review import StaleAnnotationError
from recordlinkage.review import UnknownPairError
from recordlinkage.review.generation import ReviewGeneration
from recordlinkage.review.labels import LABEL_NEGATIVE
from recordlinkage.review.labels import LABEL_POSITIVE
from recordlinkage.review.labels import LABEL_UNKNOWN
from recordlinkage.review.labels import labels_from_annotation
from recordlinkage.review.labels import normalize_key
from recordlinkage.review.sources import source_fingerprint
from recordlinkage.utils import LearningError


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------
def make_df(n, prefix):
    rng = np.random.default_rng(int(prefix.encode().hex(), 16) % 1000)
    return pd.DataFrame(
        {
            "name": [f"{prefix}name{i % 5}" for i in range(n)],
            "city": [f"city{i % 3}" for i in range(n)],
            "val": rng.normal(size=n),
        },
        index=[f"{prefix}-{i}" for i in range(n)],
    )


def block_pairs(df_a, df_b=None):
    indexer = rl.Index()
    indexer.block("city", "city")
    return indexer.index(df_a, df_b)


def build_compare():
    cmp = rl.Compare()
    cmp.string("name", "name", method="jarowinkler", label="name")
    cmp.exact("city", "city", label="city")
    cmp.numeric("val", "val", label="val")
    return cmp


@pytest.fixture
def frames():
    return make_df(20, "a"), make_df(20, "b")


@pytest.fixture
def pipeline(tmp_path, frames):
    df_a, df_b = frames
    pipe = ReviewPipeline(
        build_compare(),
        df_a,
        df_b,
        dataset_a_name="DSA",
        dataset_b_name="DSB",
        store_dir=tmp_path / "store",
    )
    pipe.set_candidates(block_pairs(df_a, df_b))
    return pipe


# ---------------------------------------------------------------------------
# Source fingerprinting / registry
# ---------------------------------------------------------------------------
class TestSourceVersioning:
    def test_fingerprint_stable_and_sensitive(self, frames):
        df_a, _ = frames
        fp1 = source_fingerprint(df_a)
        fp2 = source_fingerprint(df_a.copy())
        assert fp1 == fp2

        changed = df_a.copy()
        changed.loc[changed.index[0], "name"] = "different"
        assert source_fingerprint(changed) != fp1

        dropped = df_a.drop(df_a.index[0])
        assert source_fingerprint(dropped) != fp1

    def test_registry_dedup_uses_single_source(self):
        df = make_df(8, "d")
        registry = SourceRegistry(df, dataset_a_name="DS")
        assert registry.deduplication
        assert registry.names == ["DS"]
        assert registry.dataset_b_name == "DS"

    def test_registry_linking_requires_distinct_names(self, frames):
        df_a, df_b = frames
        with pytest.raises(ValueError):
            SourceRegistry(
                df_a, df_b, dataset_a_name="SAME", dataset_b_name="SAME"
            )

    def test_is_current_and_stale_datasets(self, frames):
        df_a, df_b = frames
        registry = SourceRegistry(
            df_a, df_b, dataset_a_name="A", dataset_b_name="B"
        )
        versions = registry.versions()
        assert registry.is_current(versions)
        assert registry.stale_datasets(versions) == []

        df_a2 = df_a.copy()
        df_a2.iloc[0, 0] = "x"
        registry2 = SourceRegistry(
            df_a2, df_b, dataset_a_name="A", dataset_b_name="B"
        )
        assert not registry2.is_current(versions)
        assert registry2.stale_datasets(versions) == ["A"]


# ---------------------------------------------------------------------------
# Pair identity: dedup direction and cross-dataset isolation
# ---------------------------------------------------------------------------
class TestPairIdentity:
    def test_dedup_reverse_is_same_pair(self):
        k1 = normalize_key("DS", "x", "DS", "y", True)
        k2 = normalize_key("DS", "y", "DS", "x", True)
        assert k1 == k2

    def test_linking_pair_is_ordered(self):
        forward = normalize_key("A", "r1", "B", "r2", False)
        reverse = normalize_key("B", "r2", "A", "r1", False)
        assert forward != reverse

    def test_same_record_id_across_datasets_no_collision(self):
        key_ab = normalize_key("A", "r1", "B", "r1", False)
        key_ac = normalize_key("A", "r1", "C", "r1", False)
        assert key_ab != key_ac

    def test_dedup_store_collapses_directions(self):
        df = make_df(10, "d")
        registry = SourceRegistry(df, dataset_a_name="DS")
        store = ReviewStore(registry)
        x, y = df.index[0], df.index[1]

        assert store.add(store.build_label("DS", x, "DS", y, 1)) == "added"
        assert store.add(store.build_label("DS", y, "DS", x, 1)) == "unchanged"
        assert len(store) == 1


# ---------------------------------------------------------------------------
# Label import rules
# ---------------------------------------------------------------------------
class TestLabelImport:
    def test_idempotent_reimport(self, pipeline):
        feats = pipeline.compute_features()
        ra, rb = feats.index[0]
        assert pipeline.label(ra, rb, 1) == "added"
        assert pipeline.label(ra, rb, 1) == "unchanged"
        assert len(pipeline.store) == 1

    def test_same_version_conflict_refused(self, pipeline):
        feats = pipeline.compute_features()
        ra, rb = feats.index[0]
        pipeline.label(ra, rb, 1)
        with pytest.raises(LabelConflictError):
            pipeline.label(ra, rb, 0)
        # original decision protected
        assert pipeline.store.labels()[0].label == LABEL_POSITIVE

    def test_unknown_placeholder_can_be_filled(self, pipeline):
        feats = pipeline.compute_features()
        ra, rb = feats.index[0]
        assert pipeline.label(ra, rb, None) == "added"
        assert pipeline.label(ra, rb, 1) == "updated"
        assert pipeline.store.labels()[0].label == LABEL_POSITIVE

    def test_unknown_record_rejected(self, pipeline):
        with pytest.raises(UnknownPairError):
            pipeline.label("does-not-exist", "a-1", 1)

    def test_stale_version_rejected(self, pipeline):
        feats = pipeline.compute_features()
        ra, rb = feats.index[0]
        stale = ReviewLabel(
            "DSA",
            ra,
            "DSB",
            rb,
            1,
            {"DSA": "0" * 64, "DSB": pipeline.registry.fingerprint("DSB")},
        )
        with pytest.raises(StaleAnnotationError):
            pipeline.store.add(stale)

    def test_new_version_replaces_old(self, pipeline):
        feats = pipeline.compute_features()
        ra, rb = feats.index[0]
        pipeline.label(ra, rb, 0)

        # Simulate a source update: new fingerprint but same record ids.
        df_a, df_b = (
            pipeline.registry.frame("DSA").copy(),
            pipeline.registry.frame("DSB").copy(),
        )
        df_a.loc[ra, "name"] = "updated"
        pipeline.refresh_sources(df_a, df_b)
        assert pipeline.label(ra, rb, 1) == "added"
        assert pipeline.store.labels()[0].label == LABEL_POSITIVE

    def test_invalid_label_value(self):
        with pytest.raises(ValueError):
            ReviewLabel("A", 1, "B", 2, 5, {"A": "x"})


# ---------------------------------------------------------------------------
# Annotation file round trip (existing format preserved)
# ---------------------------------------------------------------------------
class TestAnnotationRoundTrip:
    def test_write_then_import_annotation(self, tmp_path, frames):
        df_a, df_b = frames
        annotation_path = tmp_path / "annotation.json"
        pairs = block_pairs(df_a, df_b)[:8]
        write_annotation_file(
            annotation_path,
            pairs,
            df_a,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
        )

        # Simulate the annotator writing labels into the standard format.
        with open(annotation_path) as f:
            payload = json.load(f)
        for i, item in enumerate(payload["pairs"]):
            item["label"] = 1 if i < 4 else 0

        with open(annotation_path, "w") as f:
            json.dump(payload, f)

        pipe = ReviewPipeline(
            build_compare(),
            df_a,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
        )
        outcomes = pipe.import_annotation(annotation_path)
        assert outcomes["added"] == 8
        assert len(pipe.store.positive_labels()) == 4
        assert len(pipe.store.negative_labels()) == 4

        # Re-importing is idempotent.
        again = pipe.import_annotation(annotation_path)
        assert again["unchanged"] == 8

        # Existing read API still works on the raw file.
        result = read_annotation_file(annotation_path)
        assert result.links is not None
        assert result.distinct is not None

    def test_import_annotation_unknown_record(self, tmp_path, frames):
        df_a, df_b = frames
        annotation_path = tmp_path / "annotation.json"
        pairs = block_pairs(df_a, df_b)[:3]
        write_annotation_file(
            annotation_path,
            pairs,
            df_a,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
        )
        with open(annotation_path) as f:
            payload = json.load(f)
        payload["pairs"][0]["identifiers"]["a"]["record"] = "ghost"
        with open(annotation_path, "w") as f:
            json.dump(payload, f)

        pipe = ReviewPipeline(
            build_compare(),
            df_a,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
        )
        with pytest.raises(UnknownPairError):
            pipe.import_annotation(annotation_path)

    def test_annotation_unknown_label_is_preserved(self, tmp_path, frames):
        df_a, df_b = frames
        annotation_path = tmp_path / "annotation.json"
        pairs = block_pairs(df_a, df_b)[:4]
        write_annotation_file(
            annotation_path,
            pairs,
            df_a,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
        )
        with open(annotation_path) as f:
            payload = json.load(f)
        payload["pairs"][0]["label"] = 1
        payload["pairs"][1]["label"] = 0
        # leave the rest unlabelled (label absent -> None)
        with open(annotation_path, "w") as f:
            json.dump(payload, f)

        pipe = ReviewPipeline(
            build_compare(),
            df_a,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
        )
        pipe.import_annotation(annotation_path)
        assert len(pipe.store.positive_labels()) == 1
        assert len(pipe.store.negative_labels()) == 1
        assert len(pipe.store.unknown_labels()) == 2


# ---------------------------------------------------------------------------
# Supervised training semantics
# ---------------------------------------------------------------------------
class TestTrainingSemantics:
    def _label_balanced(self, pipeline, feats):
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipeline.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipeline.label(ra, rb, 0)
        for ra, rb in pair_list[6:8]:
            pipeline.label(ra, rb, None)

    def test_only_reviewed_pos_neg_used(self, pipeline):
        feats = pipeline.compute_features()
        self._label_balanced(pipeline, feats)
        generation = pipeline.publish()

        # The training frame contains pos + neg only, unknown excluded.
        training, match_index = pipeline._reviewed_training_frame(feats)
        assert len(training) == 6
        assert len(match_index) == 3

        md = generation.metadata()
        assert md["n_positive"] == 3
        assert md["n_negative"] == 3
        assert md["n_unknown"] == 2

    def test_unknown_alone_cannot_fit(self, pipeline):
        feats = pipeline.compute_features()
        ra, rb = feats.index[0]
        pipeline.label(ra, rb, None)
        with pytest.raises(LearningError):
            pipeline.publish()

    def test_positive_without_negative_refused(self, pipeline):
        feats = pipeline.compute_features()
        for ra, rb in list(feats.index)[:3]:
            pipeline.label(ra, rb, 1)
        with pytest.raises(LearningError):
            pipeline.publish()

    def test_all_candidates_scored(self, pipeline):
        feats = pipeline.compute_features()
        self._label_balanced(pipeline, feats)
        generation = pipeline.publish()
        assert len(generation.results) == len(feats)
        assert set(generation.results.columns) >= {"prediction", "probability"}
        assert generation.results["prediction"].isin([0, 1]).all()

    def test_probability_bounded(self, pipeline):
        feats = pipeline.compute_features()
        self._label_balanced(pipeline, feats)
        generation = pipeline.publish()
        probs = generation.probabilities
        assert probs.notna().all()
        assert ((probs >= 0) & (probs <= 1)).all()

    def test_dedup_reverse_label_still_trains(self):
        df = make_df(20, "d")
        pipe = ReviewPipeline(
            build_compare(), df, dataset_a_name="DS", store_dir=None
        )
        pipe.set_candidates(block_pairs(df))
        feats = pipe.compute_features()

        pair_list = list(feats.index)
        # Import labels with reversed orientation relative to feature rows.
        for ra, rb in pair_list[0:3]:
            pipe.label(rb, ra, 1)
        for ra, rb in pair_list[3:6]:
            pipe.label(rb, ra, 0)

        training, match_index = pipe._reviewed_training_frame(feats)
        assert len(training) == 6
        assert len(match_index) == 3
        generation = pipe.publish()
        assert len(generation.results) == len(feats)


class TestClassifierVariants:
    def _label(self, pipe, feats):
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipe.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipe.label(ra, rb, 0)

    def test_svm_has_nan_probabilities(self, frames):
        df_a, df_b = frames
        cmp = rl.Compare()
        cmp.exact("city", "city", label="city")
        pipe = ReviewPipeline(
            cmp,
            df_a,
            df_b,
            classifier=rl.SVMClassifier(),
            dataset_a_name="A",
            dataset_b_name="B",
        )
        pipe.set_candidates(block_pairs(df_a, df_b))
        feats = pipe.compute_features()
        self._label(pipe, feats)

        generation = pipe.publish()
        assert generation.probabilities.isna().all()
        assert generation.results["prediction"].isin([0, 1]).all()

    def test_classifier_hyperparameters_are_retained(self, frames):
        df_a, df_b = frames
        cmp = rl.Compare()
        cmp.exact("city", "city", label="city")
        pipe = ReviewPipeline(
            cmp,
            df_a,
            df_b,
            classifier=rl.SVMClassifier(C=0.5),
            dataset_a_name="A",
            dataset_b_name="B",
        )
        pipe.set_candidates(block_pairs(df_a, df_b))
        feats = pipe.compute_features()
        self._label(pipe, feats)
        first = pipe.publish()
        assert first.classifier.kernel.C == 0.5

        # A second generation is built from a fresh object but keeps C.
        pipe.label(*list(feats.index)[6], 1)
        second = pipe.publish()
        assert second.generation_id == first.generation_id + 1
        assert second.classifier.kernel.C == 0.5

    def test_explicit_classifier_kwargs(self, frames):
        df_a, df_b = frames
        cmp = rl.Compare()
        cmp.exact("city", "city", label="city")
        pipe = ReviewPipeline(
            cmp,
            df_a,
            df_b,
            classifier=rl.LogisticRegressionClassifier(),
            classifier_kwargs={"max_iter": 77},
            dataset_a_name="A",
            dataset_b_name="B",
        )
        pipe.set_candidates(block_pairs(df_a, df_b))
        feats = pipe.compute_features()
        self._label(pipe, feats)
        generation = pipe.publish()
        assert generation.classifier.kernel.max_iter == 77


# ---------------------------------------------------------------------------
# Generation persistence, atomicity and schema guards
# ---------------------------------------------------------------------------
class TestGenerationPersistence:
    def test_generation_artefacts_on_disk(self, tmp_path, pipeline):
        feats = pipeline.compute_features()
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipeline.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipeline.label(ra, rb, 0)
        generation = pipeline.publish()

        gen_dir = tmp_path / "store" / f"generation-{generation.generation_id}"
        assert (gen_dir / "generation.json").exists()
        assert (gen_dir / "classifier.joblib").exists()
        assert (gen_dir / "results.joblib").exists()
        assert (gen_dir / "labels.json").exists()
        assert (tmp_path / "store" / "current.json").exists()

    def test_reopen_loads_model_and_results(self, tmp_path, pipeline):
        feats = pipeline.compute_features()
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipeline.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipeline.label(ra, rb, 0)
        generation = pipeline.publish()

        df_a, df_b = (
            pipeline.registry.frame("DSA"),
            pipeline.registry.frame("DSB"),
        )
        reopened = ReviewPipeline.reopen(
            tmp_path / "store",
            build_compare(),
            df_a,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
        )
        assert reopened.current_generation_id == generation.generation_id
        pdt.assert_frame_equal(reopened.results, generation.results)
        # Reopened model predicts identically.
        pdt.assert_index_equal(
            reopened.predict(feats), generation.predict(feats)
        )

    def test_foreign_feature_schema_refused(self, pipeline):
        feats = pipeline.compute_features()
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipeline.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipeline.label(ra, rb, 0)
        pipeline.publish()

        foreign = feats.copy()
        foreign.columns = ["x", "y", "z"]
        with pytest.raises(SchemaMismatchError):
            pipeline.predict(foreign)

    def test_fit_failure_keeps_previous_generation(self, pipeline):
        feats = pipeline.compute_features()
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipeline.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipeline.label(ra, rb, 0)
        first = pipeline.publish()
        assert first.generation_id == 1

        class BrokenClassifier(rl.LogisticRegressionClassifier):
            def _fit(self, features, y=None):
                raise RuntimeError("boom")

        pipeline.classifier = BrokenClassifier()
        with pytest.raises(RuntimeError):
            pipeline.publish()

        assert ReviewGeneration.current_id(pipeline.store_dir) == 1
        # No partial generation directory left behind.
        leftover = [
            name
            for name in os.listdir(pipeline.store_dir)
            if name.startswith(".gen-")
        ]
        assert leftover == []

    def test_write_failure_keeps_previous_generation(self, pipeline, monkeypatch):
        feats = pipeline.compute_features()
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipeline.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipeline.label(ra, rb, 0)
        pipeline.publish()

        import recordlinkage.review.generation as gen_module

        def fail_dump(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(gen_module.joblib, "dump", fail_dump)
        with pytest.raises(OSError):
            pipeline.publish()
        assert ReviewGeneration.current_id(pipeline.store_dir) == 1

    def test_reopen_without_generation_raises(self, tmp_path, frames):
        df_a, df_b = frames
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(GenerationError):
            ReviewPipeline.reopen(
                empty,
                build_compare(),
                df_a,
                df_b,
                dataset_a_name="DSA",
                dataset_b_name="DSB",
            )


# ---------------------------------------------------------------------------
# Source change invalidation and feature recomputation
# ---------------------------------------------------------------------------
class TestSourceChange:
    def test_change_invalidates_labels_and_recomputes_features(self, pipeline):
        feats = pipeline.compute_features()
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipeline.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipeline.label(ra, rb, 0)
        first = pipeline.publish()

        df_a = pipeline.registry.frame("DSA").copy()
        df_b = pipeline.registry.frame("DSB")
        df_a.iloc[0, 0] = "mutated"

        invalidated = pipeline.refresh_sources(df_a, df_b)
        assert len(invalidated) == 6
        assert len(pipeline.store) == 0

        # Cached features are cleared and recomputed from current data.
        assert pipeline._features is None
        pipeline.set_candidates(block_pairs(df_a, df_b))
        new_feats = pipeline.compute_features()
        assert new_feats is not None

        # Previous generation remains usable with its own frozen results.
        assert ReviewGeneration.current_id(pipeline.store_dir) == 1
        old = ReviewGeneration.load(pipeline.store_dir, 1)
        assert len(old.results) == len(first.results)

    def test_reopen_detects_stale_generation_but_keeps_it_usable(
        self, tmp_path, frames
    ):
        df_a, df_b = frames
        store = tmp_path / "store"
        pipe = ReviewPipeline(
            build_compare(),
            df_a,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
            store_dir=store,
        )
        pipe.set_candidates(block_pairs(df_a, df_b))
        feats = pipe.compute_features()
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipe.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipe.label(ra, rb, 0)
        pipe.publish()
        assert pipe.generation_is_current

        # Reopen against changed source data.
        df_a2 = df_a.copy()
        df_a2.iloc[0, 0] = "changed"
        reopened = ReviewPipeline.reopen(
            store,
            build_compare(),
            df_a2,
            df_b,
            dataset_a_name="DSA",
            dataset_b_name="DSB",
        )
        assert reopened.current_generation_id == 1
        assert not reopened.generation_is_current
        # The frozen generation is still fully loadable and readable.
        assert len(reopened.results) == len(feats)

    def test_generations_cannot_mix_old_features(self, pipeline):
        feats = pipeline.compute_features()
        pair_list = list(feats.index)
        for ra, rb in pair_list[0:3]:
            pipeline.label(ra, rb, 1)
        for ra, rb in pair_list[3:6]:
            pipeline.label(ra, rb, 0)
        pipeline.publish()

        # Build a feature frame with a different column set.
        other_cmp = rl.Compare()
        other_cmp.exact("city", "city", label="city_only")
        other_feats = other_cmp.compute(
            pipeline._candidate_pairs,
            pipeline.registry.frame("DSA"),
            pipeline.registry.frame("DSB"),
        )
        with pytest.raises(SchemaMismatchError):
            pipeline.predict(other_feats)


# ---------------------------------------------------------------------------
# Public API stability
# ---------------------------------------------------------------------------
def test_legacy_api_unchanged():
    # Existing top-level helpers and classes are still exported.
    assert callable(rl.write_annotation_file)
    assert callable(rl.read_annotation_file)
    assert rl.LogisticRegressionClassifier is not None
    assert rl.Compare is not None
    assert rl.Index is not None


def test_label_constants():
    assert LABEL_POSITIVE == 1
    assert LABEL_NEGATIVE == 0
    assert LABEL_UNKNOWN is None


def test_labels_from_annotation_dict(frames):
    df_a, df_b = frames
    registry = SourceRegistry(
        df_a, df_b, dataset_a_name="DSA", dataset_b_name="DSB"
    )
    pairs = block_pairs(df_a, df_b)[:3]
    annotation = {
        "version": 1,
        "pairs": [
            {
                "identifiers": {
                    "a": {"dataset": "DSA", "record": ra},
                    "b": {"dataset": "DSB", "record": rb},
                },
                "label": 1,
            }
            for ra, rb in pairs
        ],
    }
    labels = labels_from_annotation(annotation, registry)
    assert len(labels) == 3
    assert all(label.label == LABEL_POSITIVE for label in labels)
    assert set(labels[0].versions) == {"DSA", "DSB"}
