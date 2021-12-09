Dataset document filters
====================

Dataset document filters filter datasets to specific documents. These are applied last, after ``dataset_code_filters``
and ``dataset_coder_filters``.

.. note:: Document filters must be able to be applied to a dataframe using only the ``document_id`` column;
    unlike the other filters, these filters are used in weighting the dataset to be representative of the
    corpus as a whole, so they need to deal with document attributes.

Defining dataset document filters
-----------------------------

Dataset document filters require a ``filter`` function that accepts the dataframe as well
as positional and keyword arguments, and returns a subset of the dataframe.

.. code:: python


Using dataset document filters
---------------------------

Dataset document filters get applied by dataset extractors, so you'll most commonly use them when
building a machine learning pipeline. They can be passed to dataset extractors using the
"document_filters" key:

.. code:: python


Built-in dataset document filters
---------------------------------

``django_lookup_filter``
*********************************

Filter the dataset to documents that match particular Django queries. Equivalent to running
``.filter(**{search_filter: search_value})`` (or ``.exclude()`` depending on whether
``exclude`` is True or False.)

.. code:: python

    "document_filters": [
        (
            "django_lookup_filter",
            [],
            {
                "search_filter": "text__iregex",
                "search_value": "disney",
                "exclude": False,
            },
        ),
    ]


``filter_by_date``
*********************************

Filter documents to those within a specific date range:

.. code:: python

    "document_filters": [
        (
            "filter_by_date",
            [],
            {
                "min_date": datetime.date(2000, 2, 1),
                "max_date": datetime.date(2000, 4, 1),
            },
        )
    ]


``filter_by_document_ids``
*********************************

Filter documents using an explicit list of document IDs (this can be useful if you want to run
a query separately and then just pass the resulting primary keys directly to the document filter):

.. code:: python

    "document_filters": [("filter_by_document_ids", [[1, 2, 3, 4, 5]], {})]



``filter_by_other_model_dataset``
*********************************

Filter the dataset using another dataset used by a LearningModel. Useful for
creating a dependency where one dataset should inherit and then filter the scope of another.
For example, the following filter would take the requested dataset, and then filter it down
to rows that were also found in the ``example_model`` ML pipeline's dataset extractor, and then
filter those rows to those that were given code "10".

.. code:: python

    "document_filters": [
        ("filter_by_other_model_dataset", ["example_model", "10"], {})
    ]


``filter_by_other_model_prediction``
*********************************

Similar to the above, but instead of filtering based off of a model's training and test data,
this instead filters based on the model's predictions. Useful if you train one model to predict
something, and then you want to dive deeper and train a more granular classifier within a certain
category. Unlike the filter above, which is by definition restricted to the scope of the data the model was
trained and evaluated on, the ``filter_by_other_model_prediction`` filter will apply itself to
whatever is in your dataframe - it doesn't use document_ids to filter, it just expects your dataset extractor to
produce a dataframe that can be passed to the classification model. So your dataset extractor has to be
comparable to / compatible with what was used to train the model, but it can be a different set of documents.

.. code:: python

    "document_filters": [
        ("filter_by_other_model_prediction", ["example_model", "10"], {})
    ]