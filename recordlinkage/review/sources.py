"""Source versioning for the human-review workflow.

A *source* is one of the data frames involved in the linkage or
deduplication problem (the left frame ``a`` and, for linking problems,
the right frame ``b``). Every source gets a content based version
fingerprint. Labels and generations bind themselves to those
fingerprints, which makes it possible to detect that the underlying data
changed and that previously accepted decisions became stale.

The fingerprint is deterministic and only depends on the frame content
and its index, so the same data always produces the same version while
any edit (a changed value, an added or removed record or column) changes
it.
"""

import hashlib
import json

import pandas


def source_fingerprint(df):
    """Compute a deterministic content fingerprint for a data frame.

    Parameters
    ----------
    df : pandas.DataFrame
        The source data frame.

    Returns
    -------
    str
        A hexadecimal SHA-256 digest of the frame content, including the
        index and column order.
    """
    if not isinstance(df, pandas.DataFrame):
        raise TypeError("expected a pandas.DataFrame as source")

    # ``to_json`` with orient="split" serialises columns, index and values
    # deterministically. ``date_format`` keeps timestamps unambiguous.
    payload = df.to_json(orient="split", date_format="iso", date_unit="ns")

    hasher = hashlib.sha256()
    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()


class SourceRegistry:
    """Registry of the named data frames and their current version.

    The registry maps a dataset name to its current fingerprint and keeps
    the frame itself around so that pairs can be validated and features
    can be recomputed.

    Parameters
    ----------
    df_a : pandas.DataFrame
        The left (or only, for deduplication) data frame.
    df_b : pandas.DataFrame, optional
        The right data frame for linking problems. ``None`` for
        deduplication.
    dataset_a_name : str, optional
        Name of the left data frame. Defaults to ``"a"``.
    dataset_b_name : str, optional
        Name of the right data frame. Defaults to ``"b"`` for linking and
        to the name of the left data frame for deduplication.
    """

    def __init__(
        self, df_a, df_b=None, dataset_a_name=None, dataset_b_name=None
    ):
        if not isinstance(df_a, pandas.DataFrame):
            raise TypeError("expected a pandas.DataFrame as df_a")

        if dataset_a_name is None:
            dataset_a_name = "a"

        if df_b is None:
            # Deduplication: both sides are the same source.
            dataset_b_name = dataset_a_name
        elif dataset_b_name is None:
            dataset_b_name = "b"

        if dataset_a_name == dataset_b_name and df_b is not None:
            raise ValueError(
                "linking two different data frames requires distinct "
                "dataset names"
            )

        self.dataset_a_name = dataset_a_name
        self.dataset_b_name = dataset_b_name
        self.deduplication = df_b is None

        self._frames = {dataset_a_name: df_a}
        self._fingerprints = {
            dataset_a_name: source_fingerprint(df_a)
        }

        if df_b is not None:
            self._frames[dataset_b_name] = df_b
            self._fingerprints[dataset_b_name] = source_fingerprint(df_b)

    @property
    def names(self):
        """list: The registered dataset names."""
        return list(self._frames.keys())

    def fingerprint(self, dataset_name):
        """Return the current fingerprint of a dataset.

        Parameters
        ----------
        dataset_name : str
            The dataset name.

        Returns
        -------
        str
            The current content fingerprint.
        """
        return self._fingerprints[dataset_name]

    def frame(self, dataset_name):
        """Return the data frame registered under ``dataset_name``."""
        return self._frames[dataset_name]

    def has_dataset(self, dataset_name):
        """Return whether ``dataset_name`` is registered."""
        return dataset_name in self._frames

    def has_record(self, dataset_name, record_id):
        """Return whether ``record_id`` is an index value of a dataset."""
        if not self.has_dataset(dataset_name):
            return False
        return record_id in self._frames[dataset_name].index

    def versions(self):
        """Return a copy of the current ``{name: fingerprint}`` mapping."""
        return dict(self._fingerprints)

    def to_dict(self):
        """Serialise the registry metadata (no frame data) to a dict."""
        return {
            "dataset_a_name": self.dataset_a_name,
            "dataset_b_name": self.dataset_b_name,
            "deduplication": self.deduplication,
            "versions": self.versions(),
        }

    @classmethod
    def from_dict(cls, d, frames):
        """Recreate a registry from serialised metadata and live frames.

        Parameters
        ----------
        d : dict
            Metadata produced by :meth:`to_dict`.
        frames : dict
            Mapping of dataset name to the current data frame.

        Returns
        -------
        SourceRegistry
            A registry whose fingerprints are recomputed from ``frames``.

        Raises
        ------
        StaleAnnotationError
            If a registered dataset is missing from ``frames``.
        """
        from recordlinkage.review.exceptions import StaleAnnotationError

        for name in d["versions"]:
            if name not in frames:
                raise StaleAnnotationError(
                    f"dataset {name!r} is no longer available"
                )

        if d["deduplication"]:
            return cls(
                frames[d["dataset_a_name"]],
                df_b=None,
                dataset_a_name=d["dataset_a_name"],
            )

        return cls(
            frames[d["dataset_a_name"]],
            frames[d["dataset_b_name"]],
            dataset_a_name=d["dataset_a_name"],
            dataset_b_name=d["dataset_b_name"],
        )

    def is_current(self, versions):
        """Check whether a stored version mapping matches current data.

        Parameters
        ----------
        versions : dict
            A ``{dataset_name: fingerprint}`` mapping as stored on a label
            or generation.

        Returns
        -------
        bool
            ``True`` when every referenced dataset still exists and still
            has the stored fingerprint.
        """
        for name, fingerprint in versions.items():
            if name not in self._fingerprints:
                return False
            if self._fingerprints[name] != fingerprint:
                return False
        return True

    def stale_datasets(self, versions):
        """Return the names of datasets that changed or were removed.

        Parameters
        ----------
        versions : dict
            A ``{dataset_name: fingerprint}`` mapping.

        Returns
        -------
        list
            The dataset names whose current fingerprint differs from the
            stored one, or that no longer exist.
        """
        stale = []
        for name, fingerprint in versions.items():
            current = self._fingerprints.get(name)
            if current != fingerprint:
                stale.append(name)
        return stale


def pair_versions(registry, dataset_a, dataset_b):
    """Collect the source versions backing a pair.

    For a deduplication pair both sides point at the same dataset, so the
    mapping contains a single entry. For a linking pair both datasets are
    included.

    Parameters
    ----------
    registry : SourceRegistry
        The source registry.
    dataset_a : str
        Dataset name of the left record.
    dataset_b : str
        Dataset name of the right record.

    Returns
    -------
    dict
        ``{dataset_name: fingerprint}`` with the versions the pair was
        decided against.
    """
    versions = {dataset_a: registry.fingerprint(dataset_a)}
    if dataset_b != dataset_a:
        versions[dataset_b] = registry.fingerprint(dataset_b)
    return versions


def versions_to_json(versions):
    """Stable JSON encoding of a version mapping."""
    return json.dumps(versions, sort_keys=True, separators=(",", ":"))
