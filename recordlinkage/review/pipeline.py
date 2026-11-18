"""End-to-end versioned human-review training and publishing pipeline.

The :class:`ReviewPipeline` ties the versioned building blocks together:

1. candidate record pairs are compared with a
   :class:`recordlinkage.Compare` to produce feature vectors;
2. human decisions are imported from annotation files and stored as
   versioned labels;
3. only the reviewed positives and negatives are fed to a supervised
   :meth:`classifier.fit` (unknown decisions are never used as
   negatives);
4. the fitted model scores *all* candidate pairs;
5. labels, feature schema, the actual fitted model and the pairwise
   results are published together as one immutable generation.

Source changes are handled explicitly: when a data frame changes, the
labels bound to its old version are invalidated and the Compare features
are recomputed against the current data, so a model is never trained on
stale labels mixed with fresh features.
"""

import os

import numpy as np
import pandas

from recordlinkage.base import BaseCompare
from recordlinkage.classifiers import LogisticRegressionClassifier
from recordlinkage.config import option_context
from recordlinkage.review.exceptions import GenerationError
from recordlinkage.review.exceptions import StaleAnnotationError
from recordlinkage.review.generation import GENERATION_PREFIX
from recordlinkage.review.generation import ReviewGeneration
from recordlinkage.review.generation import check_feature_schema
from recordlinkage.review.generation import feature_schema
from recordlinkage.review.labels import LABEL_NEGATIVE
from recordlinkage.review.labels import LABEL_POSITIVE
from recordlinkage.review.labels import ReviewStore
from recordlinkage.review.labels import labels_from_annotation
from recordlinkage.review.labels import normalize_key
from recordlinkage.review.sources import SourceRegistry
from recordlinkage.utils import LearningError


class ReviewPipeline:
    """Train and publish versioned models from human review labels.

    Parameters
    ----------
    compare : recordlinkage.base.BaseCompare
        The configured Compare object defining the feature columns.
    df_a : pandas.DataFrame
        The left (or only, for deduplication) data frame.
    df_b : pandas.DataFrame, optional
        The right data frame for linking problems.
    classifier : recordlinkage.base.BaseClassifier, optional
        A supervised classifier. Defaults to
        :class:`recordlinkage.LogisticRegressionClassifier`.
    classifier_kwargs : dict, optional
        Construction arguments used to instantiate a fresh, unfitted
        classifier for every new generation (the fitted state of the
        object passed as ``classifier`` is never reused). When omitted,
        the parameters of the sklearn kernel are reused when available.
    dataset_a_name, dataset_b_name : str, optional
        Names of the data frames. Stored on every label so record ids
        reused across datasets never collide.
    store_dir : str or os.PathLike, optional
        Directory for the persistent label store and generations. When
        given, an existing current generation is reopened automatically.
    """

    def __init__(
        self,
        compare,
        df_a,
        df_b=None,
        classifier=None,
        classifier_kwargs=None,
        dataset_a_name=None,
        dataset_b_name=None,
        store_dir=None,
    ):
        if not isinstance(compare, BaseCompare):
            raise TypeError("compare must be a recordlinkage Compare instance")

        self.compare = compare
        self.classifier = classifier or LogisticRegressionClassifier()
        self.classifier_kwargs = (
            dict(classifier_kwargs) if classifier_kwargs is not None else None
        )
        self.store_dir = None if store_dir is None else os.fspath(store_dir)

        self.registry = SourceRegistry(
            df_a,
            df_b=df_b,
            dataset_a_name=dataset_a_name,
            dataset_b_name=dataset_b_name,
        )
        self.store = ReviewStore(self.registry, path=self.store_dir)

        # Cached candidate pairs and feature vectors for the current
        # source versions.
        self._candidate_pairs = None
        self._features = None
        self._features_versions = None

        # The published generation, reopened from disk when available.
        self.generation = None
        if self.store_dir is not None:
            self._reopen_generation()

    # ------------------------------------------------------------------
    # Construction / reopening
    # ------------------------------------------------------------------
    @classmethod
    def reopen(
        cls,
        store_dir,
        compare,
        df_a,
        df_b=None,
        classifier=None,
        classifier_kwargs=None,
        dataset_a_name=None,
        dataset_b_name=None,
    ):
        """Reopen a previously persisted pipeline and its current generation.

        All arguments are the same as the constructor. The current
        generation's fitted model and pairwise results are loaded
        together; no feature or label from another generation is mixed
        in.
        """
        pipeline = cls(
            compare,
            df_a,
            df_b=df_b,
            classifier=classifier,
            classifier_kwargs=classifier_kwargs,
            dataset_a_name=dataset_a_name,
            dataset_b_name=dataset_b_name,
            store_dir=store_dir,
        )
        if pipeline.generation is None:
            raise GenerationError(
                f"no published generation found in {os.fspath(store_dir)!r}"
            )
        return pipeline

    def _reopen_generation(self):
        gen_id = ReviewGeneration.current_id(self.store_dir)
        if gen_id is not None:
            self.generation = ReviewGeneration.load(
                self.store_dir, generation_id=gen_id
            )

    @property
    def current_generation_id(self):
        """int or None: the id of the published current generation."""
        if self.generation is None:
            return None
        return self.generation.generation_id

    @property
    def generation_is_current(self):
        """bool: whether the loaded generation matches the current sources.

        ``False`` signals that a source changed after the generation was
        published: the frozen model and results are still loadable, but a
        new generation should be published from recomputed features.
        """
        if self.generation is None:
            return False
        return self.generation.is_current(self.registry)

    # ------------------------------------------------------------------
    # Source changes
    # ------------------------------------------------------------------
    def refresh_sources(self, df_a, df_b=None):
        """Register updated source frames and drop stale labels.

        The fingerprints are recomputed. Every label bound to a version
        that no longer matches is invalidated (and removed from the
        persistent store), and cached features are cleared so the next
        build recomputes them from the current data.

        Parameters
        ----------
        df_a : pandas.DataFrame
            The updated left (or only) data frame.
        df_b : pandas.DataFrame, optional
            The updated right data frame.

        Returns
        -------
        list
            The canonical pair keys whose labels were invalidated.
        """
        self.registry = SourceRegistry(
            df_a,
            df_b=df_b,
            dataset_a_name=self.registry.dataset_a_name,
            dataset_b_name=self.registry.dataset_b_name,
        )
        # Re-point the store at the new registry and purge stale labels.
        self.store.registry = self.registry
        invalidated = self.store.invalidate_stale()

        self._features = None
        self._features_versions = None
        self._candidate_pairs = None
        return invalidated

    # ------------------------------------------------------------------
    # Label import
    # ------------------------------------------------------------------
    def import_annotation(self, annotation):
        """Import a labelled annotation file under the versioning rules.

        Labels are rebound to the current source versions. Stale
        annotations (referencing changed or removed sources/records) are
        refused, same-version contradictions conflict, re-imports are
        idempotent and newer-version decisions replace older ones.

        Parameters
        ----------
        annotation : str, dict or AnnotationResult
            Path to an annotation JSON file, a parsed annotation dict or
            an :class:`recordlinkage.annotation.AnnotationResult`.

        Returns
        -------
        dict
            Counts of ``added``, ``replaced``, ``updated`` and
            ``unchanged`` labels.
        """
        if isinstance(annotation, (str, os.PathLike)):
            from recordlinkage.annotation import read_annotation_file

            annotation = read_annotation_file(annotation)

        labels = labels_from_annotation(annotation, self.registry)
        return self.store.add_many(labels)

    def label(self, record_a, record_b, decision):
        """Record a single decision for a pair.

        Parameters
        ----------
        record_a, record_b : object
            Record ids of the pair.
        decision : int or None
            ``1`` match, ``0`` distinct, ``None`` unknown.
        """
        review_label = self.store.build_label(
            self.registry.dataset_a_name,
            record_a,
            self.registry.dataset_b_name,
            record_b,
            decision,
        )
        return self.store.add(review_label)

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------
    def set_candidates(self, pairs):
        """Provide the candidate record pairs (a pandas.MultiIndex)."""
        if not isinstance(pairs, pandas.MultiIndex):
            raise TypeError("candidate pairs must be a pandas.MultiIndex")
        self._candidate_pairs = pairs
        self._features = None

    def compute_features(self, pairs=None, force=False):
        """Compute (or return cached) Compare features for the candidates.

        Features are always derived from the *current* registry frames.
        A cached frame is only reused while the source versions and
        candidate pairs are unchanged; otherwise the features are
        recomputed.

        Parameters
        ----------
        pairs : pandas.MultiIndex, optional
            Candidate pairs. Defaults to the pairs set via
            :meth:`set_candidates`.
        force : bool, optional
            Force recomputation even when the cache is still current.

        Returns
        -------
        pandas.DataFrame
            The comparison vectors.
        """
        if pairs is not None:
            self.set_candidates(pairs)

        if self._candidate_pairs is None:
            raise GenerationError(
                "no candidate pairs set; call set_candidates first"
            )

        current_versions = self.registry.versions()
        cache_valid = (
            not force
            and self._features is not None
            and self._features_versions == current_versions
        )
        if cache_valid:
            return self._features

        if self.registry.deduplication:
            features = self.compare.compute(
                self._candidate_pairs, self.registry.frame(
                    self.registry.dataset_a_name
                )
            )
        else:
            features = self.compare.compute(
                self._candidate_pairs,
                self.registry.frame(self.registry.dataset_a_name),
                self.registry.frame(self.registry.dataset_b_name),
            )

        self._features = features
        self._features_versions = current_versions
        return features

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _reviewed_training_frame(self, features):
        """Align reviewed positive/negative labels onto feature rows.

        Unknown labels are deliberately excluded, so they can never act
        as negative examples. Pair direction is normalised, which makes
        deduplication robust to ``(x, y)`` vs ``(y, x)`` ordering.
        """
        dedup = self.registry.deduplication
        ds_a = self.registry.dataset_a_name
        ds_b = self.registry.dataset_b_name

        # Map every canonical pair key to its row in the feature frame.
        key_to_pos = {}
        for pos, index_values in enumerate(features.index):
            record_a, record_b = index_values[0], index_values[1]
            key = normalize_key(ds_a, record_a, ds_b, record_b, dedup)
            key_to_pos[key] = pos

        selected = np.zeros(len(features), dtype=bool)
        match = np.zeros(len(features), dtype=bool)
        n_reviewed = 0
        n_matches = 0
        missing = []

        for review_label in self.store.labels():
            if review_label.label not in (
                LABEL_POSITIVE,
                LABEL_NEGATIVE,
            ):
                # Unknown decisions are never used for supervision.
                continue
            key = review_label.key(dedup)
            if key not in key_to_pos:
                missing.append(key)
                continue
            pos = key_to_pos[key]
            selected[pos] = True
            n_reviewed += 1
            if review_label.label == LABEL_POSITIVE:
                match[pos] = True
                n_matches += 1

        if missing:
            raise GenerationError(
                "reviewed pairs are not part of the candidate pairs and "
                f"have no features: {missing[:5]}"
            )

        if n_reviewed == 0:
            raise LearningError(
                "no reviewed positive/negative labels available to fit "
                "the classifier"
            )
        if n_matches == 0:
            raise LearningError(
                "both matches and non-matches are needed, but no reviewed "
                "positives were found"
            )
        if n_matches == n_reviewed:
            raise LearningError(
                "both matches and non-matches are needed, but no reviewed "
                "negatives were found"
            )

        training = features[selected]
        # Select matches from the training frame's own index so the
        # orientation always matches the feature rows.
        match_index = training.index[match[selected]]
        return training, match_index

    def _next_generation_id(self):
        if self.store_dir is None:
            return 1 if self.generation is None else self.generation.generation_id + 1

        existing = []
        for name in os.listdir(self.store_dir):
            if name.startswith(GENERATION_PREFIX):
                suffix = name[len(GENERATION_PREFIX):]
                try:
                    existing.append(int(suffix))
                except ValueError:
                    continue
        if not existing:
            return 1
        return max(existing) + 1

    def publish(self, pairs=None):
        """Fit on reviewed labels, score all candidates and publish a generation.

        The workflow:

        1. recompute features from the current data;
        2. fit the supervised classifier on reviewed positives and
           negatives only (unknown excluded);
        3. predict every candidate pair and, when supported, compute
           match probabilities;
        4. atomically persist labels, feature schema, the fitted model
           and pairwise results as one generation.

        If fitting or writing fails, the previously published generation
        remains the current one and stays fully usable.

        Parameters
        ----------
        pairs : pandas.MultiIndex, optional
            Candidate pairs, overriding :meth:`set_candidates`.

        Returns
        -------
        ReviewGeneration
            The newly published generation.
        """
        features = self.compute_features(pairs)

        training, match_index = self._reviewed_training_frame(features)

        # Fit first, before touching the store. A failure here leaves the
        # existing generation (and its pointer) completely untouched.
        fresh_classifier = self._new_classifier()
        fresh_classifier.fit(training, match_index)

        # Score the full candidate set, forcing a Series result so the
        # stored prediction column is stable regardless of global options.
        with option_context("classification.return_type", "series"):
            predictions = fresh_classifier.predict(features)

        probability = self._safe_probabilities(fresh_classifier, features)

        results = pandas.DataFrame(
            {
                "prediction": predictions.astype(int).values,
                "probability": probability,
            },
            index=features.index,
        )

        generation = ReviewGeneration(
            generation_id=self._next_generation_id(),
            classifier=fresh_classifier,
            feature_columns=feature_schema(features),
            labels=self.store.labels(),
            source_versions=self.registry.versions(),
            results=results,
        )

        if self.store_dir is not None:
            # Atomic save: on any error the temp directory is removed and
            # the current pointer keeps pointing at the old generation.
            generation.save(self.store_dir)

        self.generation = generation
        return generation

    def _new_classifier(self):
        """Instantiate a fresh, unfitted classifier of the configured type.

        The fitted state of the configured classifier is never reused; a
        brand new object is built for every generation. Construction
        arguments come from ``classifier_kwargs`` when given, otherwise
        the hyperparameters of the underlying sklearn kernel are cloned.
        """
        cls = type(self.classifier)

        if self.classifier_kwargs is not None:
            return cls(**self.classifier_kwargs)

        # Best-effort cloning through the sklearn kernel's parameters.
        kernel = getattr(self.classifier, "kernel", None)
        if kernel is not None and hasattr(kernel, "get_params"):
            try:
                return cls(**kernel.get_params())
            except TypeError:
                pass

        return cls()

    @staticmethod
    def _safe_probabilities(classifier, features):
        """Return match probabilities, or NaNs when the model has none."""
        try:
            with option_context("classification.return_type", "series"):
                probs = classifier.prob(features)
            return probs.values
        except AttributeError:
            # KMeans and SVM do not expose probabilities.
            return np.full(len(features), np.nan)

    # ------------------------------------------------------------------
    # Using the published generation
    # ------------------------------------------------------------------
    def _require_generation(self):
        if self.generation is None:
            raise GenerationError(
                "no generation has been published yet; call publish first"
            )
        return self.generation

    def predict(self, features=None):
        """Predict matches with the current generation.

        Foreign feature schemas are refused, so a model can never be
        applied to features from another generation.

        Parameters
        ----------
        features : pandas.DataFrame, optional
            Features to score. Defaults to the cached current features.
        """
        generation = self._require_generation()
        if features is None:
            features = self._features
        return generation.predict(features)

    def prob(self, features=None):
        """Match probabilities from the current generation."""
        generation = self._require_generation()
        if features is None:
            features = self._features
        return generation.prob(features)

    @property
    def results(self):
        """pandas.DataFrame: pairwise results of the current generation."""
        return self._require_generation().results

    @property
    def matches(self):
        """pandas.MultiIndex: pairs predicted as matches."""
        return self._require_generation().matches

    def assert_schema_matches(self, features):
        """Raise if ``features`` do not match the current generation schema."""
        generation = self._require_generation()
        check_feature_schema(features, generation.feature_columns)

    def assert_labels_current(self):
        """Raise if any stored label is bound to a changed source.

        Normally :meth:`refresh_sources` already purges these; this guard
        makes the "no stale labels mixed with fresh features" rule
        explicit and testable.
        """
        for review_label in self.store.labels():
            if not self.registry.is_current(review_label.versions):
                raise StaleAnnotationError(
                    "stale label present for pair "
                    f"{review_label.key(self.registry.deduplication)}"
                )
        return True

