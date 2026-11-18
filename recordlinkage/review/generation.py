"""Versioned, immutable generations of a trained review model.

A *generation* freezes one coherent snapshot of the supervised workflow:

* the human labels it was trained from (with their source versions);
* the **feature schema** (the ordered compare feature columns);
* the actual **fitted classifier**;
* the pairwise **predictions and probabilities** over all candidates.

All four artefacts belong to the same generation and are persisted
atomically. If fitting or writing fails, the previously published
generation stays available and loadable. Reopening a store loads that
generation's model and results together; a model can never be applied to
features or labels from another generation.
"""

import json
import os
import shutil
import tempfile

import joblib
import pandas

from recordlinkage.review.exceptions import GenerationError
from recordlinkage.review.exceptions import SchemaMismatchError

GENERATION_PREFIX = "generation-"
MODEL_FILENAME = "classifier.joblib"
RESULTS_FILENAME = "results.joblib"
METADATA_FILENAME = "generation.json"
POINTER_FILENAME = "current.json"


def feature_schema(features):
    """Extract a comparable fingerprint of a feature frame's schema.

    The schema is the ordered list of feature columns. Two feature
    frames share a schema only if the columns appear in exactly the same
    order with the same names.

    Parameters
    ----------
    features : pandas.DataFrame
        A comparison-vector frame as produced by
        :meth:`recordlinkage.Compare.compute`.

    Returns
    -------
    list
        The ordered feature column labels.
    """
    if not isinstance(features, pandas.DataFrame):
        raise TypeError("expected a pandas.DataFrame of comparison vectors")
    return [str(col) for col in features.columns]


def check_feature_schema(features, schema):
    """Raise if ``features`` does not match a stored feature schema.

    Parameters
    ----------
    features : pandas.DataFrame
        The feature frame to validate.
    schema : list
        The stored ordered feature columns.

    Raises
    ------
    SchemaMismatchError
        If the columns differ in name or order.
    """
    current = feature_schema(features)
    if current != list(schema):
        raise SchemaMismatchError(
            "feature schema mismatch: model was fitted on "
            f"{list(schema)} but received {current}"
        )


class ReviewGeneration:
    """One immutable, persistable generation of a fitted review model.

    Instances are normally created by
    :class:`recordlinkage.review.pipeline.ReviewPipeline`; low-level
    callers can also build and persist them directly.

    Parameters
    ----------
    generation_id : int
        Monotonic generation number, starting at 1.
    classifier : recordlinkage.base.BaseClassifier
        The fitted classifier.
    feature_columns : list
        Ordered feature columns the classifier was fitted on.
    labels : list of ReviewLabel
        The labels used for training.
    source_versions : dict
        The ``{dataset: fingerprint}`` mapping current at build time.
    results : pandas.DataFrame
        Pairwise predictions and probabilities indexed by the candidate
        pair MultiIndex.
    """

    def __init__(
        self,
        generation_id,
        classifier,
        feature_columns,
        labels,
        source_versions,
        results,
    ):
        self.generation_id = generation_id
        self.classifier = classifier
        self.feature_columns = list(feature_columns)
        self.labels = list(labels)
        self.source_versions = dict(source_versions)
        self.results = results

    def is_current(self, registry):
        """Return whether this generation matches the current sources.

        Parameters
        ----------
        registry : recordlinkage.review.SourceRegistry
            The current source registry.

        Returns
        -------
        bool
            ``True`` when every source version frozen in this generation
            still equals the current fingerprint.
        """
        return registry.is_current(self.source_versions)

    # ------------------------------------------------------------------
    # Views on the frozen results
    # ------------------------------------------------------------------
    @property
    def matches(self):
        """pandas.MultiIndex: candidate pairs predicted as matches."""
        if "prediction" not in self.results.columns:
            return None
        mask = self.results["prediction"] == 1
        return self.results.index[mask]

    @property
    def predictions(self):
        """pandas.Series: the 0/1 prediction for every candidate pair."""
        return self.results["prediction"]

    @property
    def probabilities(self):
        """pandas.Series or None: match probability, when supported."""
        if "probability" in self.results.columns:
            return self.results["probability"]
        return None

    def metadata(self):
        """Return the JSON-serialisable metadata describing this generation."""
        return {
            "generation_id": self.generation_id,
            "classifier_class": (
                f"{type(self.classifier).__module__}."
                f"{type(self.classifier).__name__}"
            ),
            "feature_schema": self.feature_columns,
            "source_versions": self.source_versions,
            "n_labels": len(self.labels),
            "n_positive": sum(label.label == 1 for label in self.labels),
            "n_negative": sum(label.label == 0 for label in self.labels),
            "n_unknown": sum(label.label is None for label in self.labels),
            "n_candidates": len(self.results),
            "columns": list(self.results.columns),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _dir(self, store_path):
        return os.path.join(
            store_path, f"{GENERATION_PREFIX}{self.generation_id}"
        )

    def save(self, store_path):
        """Persist the generation atomically and point ``current`` at it.

        The generation is written to a temporary directory first. Only
        once every artefact (metadata, model, results, labels) is on disk
        does the directory get renamed into place and the ``current``
        pointer updated. A failure during fitting/writing therefore leaves
        the previous generation (and its pointer) untouched.

        Parameters
        ----------
        store_path : str or os.PathLike
            The review store directory.
        """
        store_path = os.fspath(store_path)
        os.makedirs(store_path, exist_ok=True)

        target = self._dir(store_path)
        if os.path.exists(target):
            raise GenerationError(
                f"generation {self.generation_id} already exists at {target}"
            )

        tmp = tempfile.mkdtemp(prefix=".gen-", dir=store_path)
        try:
            # 1. metadata
            with open(
                os.path.join(tmp, METADATA_FILENAME), "w", encoding="utf-8"
            ) as f:
                json.dump(self.metadata(), f, indent=2)

            # 2. labels (the exact decisions this generation trained on)
            with open(
                os.path.join(tmp, "labels.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "labels": [label.to_dict() for label in self.labels]
                    },
                    f,
                    indent=2,
                )

            # 3. the actual fitted classifier
            joblib.dump(
                self.classifier,
                os.path.join(tmp, MODEL_FILENAME),
                compress=3,
            )

            # 4. pairwise predictions / probabilities
            joblib.dump(
                self.results,
                os.path.join(tmp, RESULTS_FILENAME),
                compress=3,
            )

            # Move the completed generation into place, then flip the
            # pointer atomically.
            os.replace(tmp, target)
            self._write_pointer(store_path)
        except Exception:
            if os.path.exists(tmp):
                shutil.rmtree(tmp, ignore_errors=True)
            raise

    def _write_pointer(self, store_path):
        pointer = os.path.join(store_path, POINTER_FILENAME)
        fd, tmp = tempfile.mkstemp(dir=store_path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {"current_generation": self.generation_id}, f, indent=2
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, pointer)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    @classmethod
    def current_id(cls, store_path):
        """Return the current generation id, or ``None`` if none published."""
        pointer = os.path.join(os.fspath(store_path), POINTER_FILENAME)
        if not os.path.exists(pointer):
            return None
        with open(pointer, encoding="utf-8") as f:
            return json.load(f)["current_generation"]

    @classmethod
    def load(cls, store_path, generation_id=None):
        """Load a generation, defaulting to the current one.

        Parameters
        ----------
        store_path : str or os.PathLike
            The review store directory.
        generation_id : int, optional
            Generation to load. Defaults to the published current
            generation.

        Returns
        -------
        ReviewGeneration

        Raises
        ------
        GenerationError
            If no generation exists or the requested one is missing.
        """
        store_path = os.fspath(store_path)

        if generation_id is None:
            generation_id = cls.current_id(store_path)
            if generation_id is None:
                raise GenerationError(
                    f"no published generation found in {store_path!r}"
                )

        gen_dir = os.path.join(
            store_path, f"{GENERATION_PREFIX}{generation_id}"
        )
        if not os.path.isdir(gen_dir):
            raise GenerationError(
                f"generation {generation_id} not found at {gen_dir!r}"
            )

        with open(
            os.path.join(gen_dir, METADATA_FILENAME), encoding="utf-8"
        ) as f:
            metadata = json.load(f)

        with open(
            os.path.join(gen_dir, "labels.json"), encoding="utf-8"
        ) as f:
            label_payload = json.load(f)

        # Import locally to avoid a circular import at module load time.
        from recordlinkage.review.labels import ReviewLabel

        labels = [
            ReviewLabel.from_dict(raw) for raw in label_payload["labels"]
        ]

        classifier = joblib.load(os.path.join(gen_dir, MODEL_FILENAME))
        results = joblib.load(
            os.path.join(gen_dir, RESULTS_FILENAME)
        )

        return cls(
            generation_id=metadata["generation_id"],
            classifier=classifier,
            feature_columns=metadata["feature_schema"],
            labels=labels,
            source_versions=metadata["source_versions"],
            results=results,
        )

    def predict(self, features):
        """Predict with this generation, refusing foreign feature schemas.

        Parameters
        ----------
        features : pandas.DataFrame
            Comparison vectors. Must use the exact feature schema this
            generation was trained on.

        Returns
        -------
        pandas.MultiIndex
            The pairs predicted as matches.
        """
        check_feature_schema(features, self.feature_columns)
        return self.classifier.predict(features)

    def prob(self, features):
        """Match probabilities, refusing foreign feature schemas."""
        check_feature_schema(features, self.feature_columns)
        return self.classifier.prob(features)
