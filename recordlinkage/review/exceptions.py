"""Exceptions for the versioned human-review workflow.

All errors raised by the :mod:`recordlinkage.review` package derive from
:class:`ReviewError` so callers can catch the whole family with a single
``except ReviewError`` clause.
"""


class ReviewError(Exception):
    """Base class for every error raised by :mod:`recordlinkage.review`."""


class StaleAnnotationError(ReviewError):
    """Raised when an annotation refers to an outdated or removed source.

    The annotation was rendered against a source version that no longer
    matches the registered data. The labels in it are stale and cannot be
    imported.
    """


class LabelConflictError(ReviewError):
    """Raised when an imported label contradicts an accepted decision.

    Contradictions come in two flavours:

    * the exact same source version already carries a *different* decision
      (same version, different value);
    * an annotation tries to re-open a pair that already has an accepted
      decision for that same source version (old decision).

    A label bound to a *newer* source version is never a conflict: it
    replaces the previous decision.
    """


class UnknownPairError(ReviewError):
    """Raised when a pair references records that do not exist in a source."""


class GenerationError(ReviewError):
    """Raised when a generation cannot be built or loaded."""


class SchemaMismatchError(GenerationError):
    """Raised when features or labels are mixed across generations.

    A fitted model is only valid for the exact feature schema it was
    trained on and the label generation it was trained with. Applying a
    model to features with another schema, or loading results from another
    generation, is refused.
    """
