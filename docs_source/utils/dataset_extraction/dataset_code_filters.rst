Dataset code filters
====================

Dataset code filters filter datasets to specific code values. These are applied first, followed by
``dataset_coder_filters`` and then ``dataset_document_filters``.

Defining dataset code filters
-----------------------------

Dataset code filters require a ``filter`` function that accepts the dataframe as well
as positional and keyword arguments, and returns a subset of the dataframe.
Here's an example of one that makes sure filters documents to a subset of code values we pass in as arguments:

.. code:: python

    from django_pewtils import get_model

    def filter(self, df, *args, **kwargs):
        pks = (
            get_model("Question", app_name="django_learning")
                .objects.get(name="test_checkbox")
                .labels.filter(value__in=args)
                .values_list("pk", flat=True)
        )
        return df[df["label_id"].isin(pks)]


Using dataset code filters
---------------------------

Dataset code filters get applied by dataset extractors, so you'll most commonly use them when
building a machine learning pipeline. They can be passed to dataset extractors using the
"code_filters" key. For each code filter to apply, you need to pass it a tuple with its name,
positional arguments, and keyword arguments:

.. code:: python

    {
        "dataset_extractor": {
            "name": "document_dataset",
            "parameters": {
                "project_name": "my_project",
                "sample_names": ["my_sample"],
                "question_names": ["test_checkbox"],
                "code_filters": [
                    ("my_code_filter", ["1", "2"], {})
                ]  # goes here
            },
            "outcome_column": "label_id",
        },
        "model": { ... }
    }
