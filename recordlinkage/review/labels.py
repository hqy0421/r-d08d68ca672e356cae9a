"""Versioned human-review labels.

This module holds the label model and the persistent label store used by
the human-review workflow.

A label binds a decision to:

* the **pair** of records, identified by both dataset name *and* record
  id on each side, so the same record id in two different datasets can
  never be confused;
* the **source versions** (content fingerprints) against which the
  decision was made.

Decisions are explicit: ``1`` (a link / match), ``0`` (distinct) or
``None`` (unknown / not decided yet). Unknown is a first-class state,
*not* a negative example: it is stored but never handed to a supervised
classifier as a non-match.
"""

import json
import os
import tempfile

import pandas

from recordlinkage.review.exceptions import LabelConflictError
from recordlinkage.review.exceptions import StaleAnnotationError
from recordlinkage.review.exceptions import UnknownPairError
from recordlinkage.review.sources import pair_versions
from recordlinkage.review.sources import versions_to_json

# The three accepted decisions. ``None`` represents "unknown".
LABEL_POSITIVE = 1
LABEL_NEGATIVE = 0
LABEL_UNKNOWN = None

_VALID_LABELS = (1, 0, None)

LABELS_FILENAME = "labels.json"


def normalize_key(dataset_a, record_a, dataset_b, record_b, deduplication):
    """Normalise a pair identity into a canonical key.

    For deduplication, ``(x, y)`` and ``(y, x)`` describe the *same*
    unordered pair. The two directions are folded onto a single key by
    sorting the record ids. For linking, the pair is ordered because the
    two sides are different datasets.

    The dataset names are part of the key, which guarantees that record
    ids reused across two datasets are never mixed up.

    Parameters
    ----------
    dataset_a, dataset_b : str
        Dataset names of the two sides.
    record_a, record_b : object
        Record ids (index values) of the two sides.
    deduplication : bool
        Whether the pair lives inside a single dataset.

    Returns
    -------
    tuple
        A hashable canonical key
        ``(dataset_a, record_a, dataset_b, record_b)``.
    """
    if deduplication:
        if dataset_a != dataset_b:
            raise ValueError(
                "deduplication pairs must reference a single dataset"
            )
        # Fold the two directions of the same unordered pair together.
        left, right = sorted((record_a, record_b), key=_sort_key)
        return (dataset_a, left, dataset_a, right)

    return (dataset_a, record_a, dataset_b, record_b)


def _sort_key(value):
    """Sorting key that works for mixed-type record ids."""
    # Numbers sort before strings; this avoids TypeError on mixed ids
    # while keeping deterministic order within a single type.
    return (type(value).__name__, str(value))


class ReviewLabel:
    """A single human decision on a record pair.

    Parameters
    ----------
    dataset_a, dataset_b : str
        Dataset names of the two sides.
    record_a, record_b : object
        Record ids (index values) of the two sides.
    label : int or None
        ``1`` for a match, ``0`` for distinct, ``None`` for unknown.
    versions : dict
        ``{dataset_name: fingerprint}`` the decision was made against.
    """

    def __init__(
        self, dataset_a, record_a, dataset_b, record_b, label, versions
    ):
        if label not in _VALID_LABELS:
            raise ValueError(
                f"label must be one of {_VALID_LABELS}, got {label!r}"
            )
        self.dataset_a = dataset_a
        self.record_a = record_a
        self.dataset_b = dataset_b
        self.record_b = record_b
        self.label = label
        self.versions = dict(versions)

    def key(self, deduplication):
        """Return the canonical pair key."""
        return normalize_key(
            self.dataset_a,
            self.record_a,
            self.dataset_b,
            self.record_b,
            deduplication,
        )

    def to_dict(self):
        """Serialise the label to a JSON-friendly dict."""
        return {
            "identifiers": {
                "a": {
                    "dataset": self.dataset_a,
                    "record": self.record_a,
                },
                "b": {
                    "dataset": self.dataset_b,
                    "record": self.record_b,
                },
            },
            "label": self.label,
            "versions": self.versions,
        }

    @classmethod
    def from_dict(cls, d):
        """Recreate a label from a serialised dict."""
        ids = d["identifiers"]
        return cls(
            ids["a"]["dataset"],
            ids["a"]["record"],
            ids["b"]["dataset"],
            ids["b"]["record"],
            d.get("label"),
            d.get("versions", {}),
        )

    def __repr__(self):
        return (
            f"<ReviewLabel ({self.dataset_a}:{self.record_a!r}, "
            f"{self.dataset_b}:{self.record_b!r}) label={self.label!r}>"
        )


class ReviewStore:
    """Persistent, version-aware store of human review labels.

    The store keeps every accepted decision and enforces the import
    rules:

    * importing a label whose source versions are stale (a source
      changed or was removed) is refused;
    * importing a pair referencing unknown records is refused;
    * re-importing the *same* decision for the *same* versions is
      idempotent (no change, no error);
    * a different decision for the *same* versions conflicts and is
      refused (old decisions are protected);
    * a decision bound to *new* versions replaces the previous one.

    Parameters
    ----------
    registry : SourceRegistry
        The registry describing the current source data.
    path : str or os.PathLike, optional
        Directory in which the store persists ``labels.json``. When
        ``None`` the store is in-memory only.
    """

    def __init__(self, registry, path=None):
        self.registry = registry
        self.path = None if path is None else os.fspath(path)
        # Mapping of canonical key -> ReviewLabel
        self._labels = {}
        if self.path is not None:
            os.makedirs(self.path, exist_ok=True)
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @property
    def _file(self):
        return os.path.join(self.path, LABELS_FILENAME)

    def _load(self):
        if not os.path.exists(self._file):
            return
        with open(self._file, encoding="utf-8") as f:
            content = json.load(f)

        for raw in content.get("labels", []):
            label = ReviewLabel.from_dict(raw)
            key = label.key(self.registry.deduplication)
            self._labels[key] = label

    def _save(self):
        """Atomically persist the labels, keeping the file valid on failure."""
        if self.path is None:
            return

        payload = {
            "registry": self.registry.to_dict(),
            "labels": [label.to_dict() for label in self._labels.values()],
        }

        # Write to a temp file in the same directory then atomically
        # replace, so a crash never leaves a truncated labels file.
        fd, tmp = tempfile.mkstemp(dir=self.path, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self._file)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._labels)

    def __contains__(self, key):
        return key in self._labels

    def get(self, key):
        """Return the stored :class:`ReviewLabel` for ``key`` or ``None``."""
        return self._labels.get(key)

    def labels(self):
        """Return all stored labels as a list (insertion order)."""
        return list(self._labels.values())

    def _labels_with_state(self, state):
        return [
            label for label in self._labels.values() if label.label == state
        ]

    def positive_labels(self):
        """Return the labels decided as matches (``label == 1``)."""
        return self._labels_with_state(LABEL_POSITIVE)

    def negative_labels(self):
        """Return the labels decided as distinct (``label == 0``)."""
        return self._labels_with_state(LABEL_NEGATIVE)

    def unknown_labels(self):
        """Return the labels marked unknown (``label is None``)."""
        return self._labels_with_state(LABEL_UNKNOWN)

    def _to_multiindex(self, labels):
        if not labels:
            return None
        tuples = [
            (label.record_a, label.record_b) for label in labels
        ]
        return pandas.MultiIndex.from_tuples(tuples)

    def links(self):
        """Return the decided matches as a :class:`pandas.MultiIndex`."""
        return self._to_multiindex(self.positive_labels())

    def distinct(self):
        """Return the decided distinct pairs as a MultiIndex."""
        return self._to_multiindex(self.negative_labels())

    def unknown(self):
        """Return the unknown pairs as a MultiIndex."""
        return self._to_multiindex(self.unknown_labels())

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def _validate_pair(self, label):
        """Check that the pair and its versions are still valid."""
        # Both referenced datasets must still exist.
        for dataset_name in (label.dataset_a, label.dataset_b):
            if not self.registry.has_dataset(dataset_name):
                raise StaleAnnotationError(
                    f"dataset {dataset_name!r} was removed; label for "
                    f"({label.record_a!r}, {label.record_b!r}) is stale"
                )

        # Both records must still be present.
        if not self.registry.has_record(label.dataset_a, label.record_a):
            raise UnknownPairError(
                f"record {label.record_a!r} no longer exists in dataset "
                f"{label.dataset_a!r}"
            )
        if not self.registry.has_record(label.dataset_b, label.record_b):
            raise UnknownPairError(
                f"record {label.record_b!r} no longer exists in dataset "
                f"{label.dataset_b!r}"
            )

        # The label must describe the *current* version of every source.
        stale = self.registry.stale_datasets(label.versions)
        if stale:
            raise StaleAnnotationError(
                "label for pair "
                f"({label.dataset_a}:{label.record_a!r}, "
                f"{label.dataset_b}:{label.record_b!r}) was made against "
                f"an outdated version of {stale}; re-annotate against the "
                "current data"
            )

    def add(self, label, force=False):
        """Add or update a label under the import rules.

        Parameters
        ----------
        label : ReviewLabel
            The decision to import.
        force : bool, optional
            Internal escape hatch that skips the staleness/conflict
            checks. Application code should leave this ``False``.

        Returns
        -------
        str
            One of ``"added"`` (new pair), ``"replaced"`` (new source
            version overwrote an older decision), ``"updated"`` (an
            unknown placeholder became a real decision on the same
            version) or ``"unchanged"`` (identical re-import).
        """
        key = label.key(self.registry.deduplication)

        if not force:
            self._validate_pair(label)

        existing = self._labels.get(key)

        if existing is None:
            self._labels[key] = label
            self._save()
            return "added"

        same_versions = versions_to_json(existing.versions) == versions_to_json(
            label.versions
        )

        if same_versions:
            if existing.label == label.label:
                # Idempotent re-import.
                return "unchanged"

            # An unknown placeholder may be filled in; any other change
            # of value on the exact same versions is a hard conflict.
            if existing.label is LABEL_UNKNOWN or label.label is LABEL_UNKNOWN:
                self._labels[key] = label
                self._save()
                return "updated"

            raise LabelConflictError(
                "conflicting decision for pair "
                f"{key} on the same source version: existing "
                f"label={existing.label!r}, imported label={label.label!r}"
            )

        # Different versions. Only a newer/current version may replace an
        # older one, and _validate_pair already proved the incoming
        # versions equal the current registry state. Refuse a regression
        # to an older version.
        if not force and not self.registry.is_current(label.versions):
            raise StaleAnnotationError(
                f"cannot replace the current decision for pair {key} with "
                "a label bound to an older source version"
            )

        self._labels[key] = label
        self._save()
        return "replaced"

    def add_many(self, labels, force=False):
        """Import several labels transactionally.

        All labels are validated first. If any single label is rejected,
        nothing is written.

        Returns
        -------
        dict
            Counts keyed by outcome (``added``, ``replaced``,
            ``updated``, ``unchanged``).
        """
        # Validate everything before mutating anything.
        prepared = []
        for label in labels:
            key = label.key(self.registry.deduplication)
            if not force:
                self._validate_pair(label)
            prepared.append((key, label))

        # Pre-compute conflict checks against a virtual merged state so
        # that two contradictory labels inside the same batch are also
        # caught.
        virtual = dict(self._labels)
        outcomes = {
            "added": 0,
            "replaced": 0,
            "updated": 0,
            "unchanged": 0,
        }
        for key, label in prepared:
            existing = virtual.get(key)
            if existing is None:
                virtual[key] = label
                outcomes["added"] += 1
                continue

            same_versions = versions_to_json(
                existing.versions
            ) == versions_to_json(label.versions)
            if same_versions:
                if existing.label == label.label:
                    outcomes["unchanged"] += 1
                    continue
                if (
                    existing.label is LABEL_UNKNOWN
                    or label.label is LABEL_UNKNOWN
                ):
                    virtual[key] = label
                    outcomes["updated"] += 1
                    continue
                raise LabelConflictError(
                    "conflicting decision for pair "
                    f"{key} on the same source version: existing "
                    f"label={existing.label!r}, imported "
                    f"label={label.label!r}"
                )

            if not force and not self.registry.is_current(label.versions):
                raise StaleAnnotationError(
                    "cannot replace the current decision for pair "
                    f"{key} with an older source version"
                )
            virtual[key] = label
            outcomes["replaced"] += 1

        # Commit.
        self._labels = virtual
        self._save()
        return outcomes

    def invalidate_stale(self):
        """Drop labels whose sources changed or were removed.

        After a source update, decisions bound to the old version are no
        longer trustworthy. This method removes them and returns the
        affected canonical keys so callers can re-annotate.

        Returns
        -------
        list
            The canonical keys that were invalidated.
        """
        invalid = [
            key
            for key, label in self._labels.items()
            if not self.registry.is_current(label.versions)
        ]
        for key in invalid:
            del self._labels[key]
        if invalid:
            self._save()
        return invalid

    # ------------------------------------------------------------------
    # Annotation (re-)export
    # ------------------------------------------------------------------
    def build_label(self, dataset_a, record_a, dataset_b, record_b, label):
        """Construct a :class:`ReviewLabel` bound to current versions.

        Convenience helper that fills in the source versions from the
        registry automatically.
        """
        versions = pair_versions(self.registry, dataset_a, dataset_b)
        return ReviewLabel(
            dataset_a,
            record_a,
            dataset_b,
            record_b,
            label,
            versions,
        )


def iter_annotation_pairs(annotation):
    """Yield ``(identifiers, label)`` triples from an annotation object.

    Parameters
    ----------
    annotation : dict or AnnotationResult
        An annotation file parsed as a dict, or an
        :class:`recordlinkage.annotation.AnnotationResult`.

    Yields
    ------
    tuple
        ``(dataset_a, record_a, dataset_b, record_b, label)`` for every
        pair in the annotation.
    """
    if isinstance(annotation, dict):
        pairs = annotation["pairs"]
    else:
        # AnnotationResult exposes the raw pair dicts.
        pairs = annotation.pairs

    for item in pairs:
        ids = item["identifiers"]
        yield (
            ids["a"]["dataset"],
            ids["a"]["record"],
            ids["b"]["dataset"],
            ids["b"]["record"],
            item.get("label"),
        )


def labels_from_annotation(annotation, registry):
    """Convert an annotation into registry-versioned :class:`ReviewLabel`.

    Every pair is rebound to the *current* fingerprint of the datasets
    it references. Pairs referencing unknown datasets or records raise
    immediately.

    Parameters
    ----------
    annotation : dict or AnnotationResult
        The imported annotation.
    registry : SourceRegistry
        The current source registry.

    Returns
    -------
    list of ReviewLabel

    Raises
    ------
    StaleAnnotationError
        If a pair references a dataset that is not registered.
    UnknownPairError
        If a pair references a record id that does not exist.
    """
    labels = []
    for dataset_a, record_a, dataset_b, record_b, label in iter_annotation_pairs(
        annotation
    ):
        for dataset_name in (dataset_a, dataset_b):
            if not registry.has_dataset(dataset_name):
                raise StaleAnnotationError(
                    f"annotation references unknown dataset {dataset_name!r}"
                )
        if not registry.has_record(dataset_a, record_a):
            raise UnknownPairError(
                f"annotation references record {record_a!r} that does not "
                f"exist in dataset {dataset_a!r}"
            )
        if not registry.has_record(dataset_b, record_b):
            raise UnknownPairError(
                f"annotation references record {record_b!r} that does not "
                f"exist in dataset {dataset_b!r}"
            )

        versions = pair_versions(registry, dataset_a, dataset_b)
        labels.append(
            ReviewLabel(
                dataset_a,
                record_a,
                dataset_b,
                record_b,
                label,
                versions,
            )
        )
    return labels
