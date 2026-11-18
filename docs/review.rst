*******************************************
Versioned human-review training and release
*******************************************

Manually labelled pairs are valuable but fragile: as soon as the source
data changes, a decision made against the old data may no longer be
correct. The :mod:`recordlinkage.review` package adds a versioned,
human-in-the-loop training and release process on top of the existing
:doc:`annotation` workflow. It keeps the annotation file format and the
Compare/classifier APIs unchanged.

Concepts
========

* A **source** is one of the data frames in the problem. Every source
  gets a content fingerprint (:func:`recordlinkage.review.source_fingerprint`),
  so any change to values, records or columns produces a new version.
* A **label** binds an explicit decision -- positive (``1``), negative
  (``0``) or unknown (``None``) -- to a pair of records *and* to the
  source versions the decision was made against. The pair identity
  contains the dataset name on both sides, so the same record id in two
  datasets is never confused.
* A **generation** freezes one coherent, immutable release: the labels,
  the feature schema, the actual fitted classifier and the pairwise
  predictions/probabilities. These four artefacts are persisted together.

Import rules
============

Labels are imported from ordinary annotation files through
:meth:`recordlinkage.review.ReviewPipeline.import_annotation`. The store
enforces the following rules:

* an annotation referencing a changed or removed source, or a missing
  record, is rejected (``StaleAnnotationError`` / ``UnknownPairError``);
* re-importing the same decision for the same version is idempotent;
* a different decision on the *same* source version conflicts and is
  rejected (``LabelConflictError``); old decisions are protected;
* a decision bound to a *new* source version replaces the previous one;
* for deduplication, ``(x, y)`` and ``(y, x)`` are the same unordered
  pair, so importing either direction is recognised as the same label.

Unknown decisions are stored but **never** used as negative examples.
Only reviewed positives and negatives are passed to the supervised
``classifier.fit``.

A complete workflow
===================

.. code:: python

    import recordlinkage as rl
    from recordlinkage.review import ReviewPipeline

    # 1. Configure the compare features as usual.
    compare = rl.Compare()
    compare.string("name", "name", method="jarowinkler", label="name")
    compare.exact("city", "city", label="city")

    # 2. Create the pipeline. It fingerprints the sources and, when a
    #    store directory is given, reopens the current generation.
    pipeline = ReviewPipeline(
        compare,
        df_a,
        df_b,
        classifier=rl.LogisticRegressionClassifier(),
        dataset_a_name="customers",
        dataset_b_name="leads",
        store_dir="./review_store",
    )

    # 3. Set the candidate pairs and import the human annotations.
    indexer = rl.Index()
    indexer.block("city", "city")
    pipeline.set_candidates(indexer.index(df_a, df_b))
    pipeline.import_annotation("labelled_pairs.json")

    # 4. Fit on reviewed pos/neg only and score every candidate pair.
    generation = pipeline.publish()
    print(generation.results)          # prediction + probability per pair
    print(generation.matches)          # MultiIndex of predicted matches

Reopening after a restart loads that generation's model and results
together:

.. code:: python

    pipeline = ReviewPipeline.reopen(
        "./review_store", compare, df_a, df_b,
        dataset_a_name="customers", dataset_b_name="leads",
    )
    pipeline.predict(features)

Handling source changes
=======================

When a data frame changes, call
:meth:`recordlinkage.review.ReviewPipeline.refresh_sources`. Labels bound
to the old version are invalidated and removed, cached features are
cleared and the next :meth:`publish` recomputes the Compare features from
the current data. The previously published generation stays available and
loadable; a model can never be applied to features with a different
schema, which raises ``SchemaMismatchError``.

If fitting or writing a generation fails, the temporary artefacts are
discarded and the previous generation remains the current one, so a
broken training run never takes down a working release.

Reference
=========

.. autoclass:: recordlinkage.review.ReviewPipeline
    :members:

.. autoclass:: recordlinkage.review.ReviewGeneration
    :members:

.. autoclass:: recordlinkage.review.ReviewStore
    :members:

.. autoclass:: recordlinkage.review.SourceRegistry
    :members:

.. autofunction:: recordlinkage.review.source_fingerprint

.. autoexception:: recordlinkage.review.StaleAnnotationError
.. autoexception:: recordlinkage.review.LabelConflictError
.. autoexception:: recordlinkage.review.SchemaMismatchError
