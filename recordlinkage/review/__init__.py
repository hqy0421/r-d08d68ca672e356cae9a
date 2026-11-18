"""Versioned human-review training and publishing.

This package layers a versioned, human-in-the-loop training and release
process on top of the existing record linkage API:

* :class:`ReviewPipeline` is the main entry point;
* :class:`SourceRegistry` fingerprints the source data frames so source
  changes are detected;
* :class:`ReviewStore` / :class:`ReviewLabel` persist decisions bound to
  source versions;
* :class:`ReviewGeneration` freezes labels, feature schema, the fitted
  classifier and pairwise results into one immutable generation.

The original annotation format (see :mod:`recordlinkage.annotation`) and
the classifier/compare APIs are unchanged.
"""

from recordlinkage.review.exceptions import GenerationError
from recordlinkage.review.exceptions import LabelConflictError
from recordlinkage.review.exceptions import ReviewError
from recordlinkage.review.exceptions import SchemaMismatchError
from recordlinkage.review.exceptions import StaleAnnotationError
from recordlinkage.review.exceptions import UnknownPairError
from recordlinkage.review.generation import ReviewGeneration
from recordlinkage.review.labels import LABEL_NEGATIVE
from recordlinkage.review.labels import LABEL_POSITIVE
from recordlinkage.review.labels import LABEL_UNKNOWN
from recordlinkage.review.labels import ReviewLabel
from recordlinkage.review.labels import ReviewStore
from recordlinkage.review.labels import labels_from_annotation
from recordlinkage.review.pipeline import ReviewPipeline
from recordlinkage.review.sources import SourceRegistry
from recordlinkage.review.sources import source_fingerprint

__all__ = [
    "ReviewPipeline",
    "ReviewGeneration",
    "ReviewStore",
    "ReviewLabel",
    "SourceRegistry",
    "source_fingerprint",
    "labels_from_annotation",
    "LABEL_POSITIVE",
    "LABEL_NEGATIVE",
    "LABEL_UNKNOWN",
    "ReviewError",
    "StaleAnnotationError",
    "LabelConflictError",
    "UnknownPairError",
    "GenerationError",
    "SchemaMismatchError",
]
