Dataset coder filters
====================

Dataset coder filters filter datasets to specific coders and/or documents based on coder attributes.
These are applied second, after ``dataset_code_filters`` and before ``dataset_document_filters``.

Defining dataset coder filters
-----------------------------

Dataset coder filters require a ``filter`` function that accepts the dataframe as well
as positional and keyword arguments, and returns a subset of the dataframe. Here's an
example of one that filters to coders whose names start with "a", for whatever reason:

.. code:: python

    def filter(self, df, coder_names, **kwargs):
        return df[df["coder_name"].str.startswith("a")]


Using dataset coder filters
---------------------------

Dataset coder filters get applied by dataset extractors, so you'll most commonly use them when
building a machine learning pipeline. They can be passed to dataset extractors using the
"coder_filters" key:

.. code:: python

    {
        "dataset_extractor": {
            "name": "document_dataset",
            "parameters": {
                "project_name": "my_project",
                "sample_names": ["my_sample"],
                "question_names": ["test_checkbox"],
                "coder_filters": [
                    ("exclude_mturk", [], {})
                ]  # goes here
            },
            "outcome_column": "label_id",
        },
        "model": { ... }
    }

Built-in dataset coder filters
-------------------------------

``exclude_by_coder_names``
*********************************

Exclude specific coders by name, passed in as positional arguments:

.. code:: python

    "coder_filters": [
        ("exclude_by_coder_names", ["coder1", "coder2"], {})
    ]


``exclude_experts``
*********************************

Exclude in-house coders (keeping only Mechanical Turk results):

.. code:: python

    "coder_filters": [
        ("exclude_experts", [], {})
    ]


``exclude_mturk``
*********************************

Exclude Mechancial Turk coders, keeping only in-house results:

.. code:: python

    "coder_filters": [
        ("exclude_mturk", [], {})
    ]


``filter_by_coder_names``
*********************************

Filter to specific coders by name, passed in as positional arguments:

.. code:: python

    "coder_filters": [
        ("filter_by_coder_names", ["coder1", "coder2"], {})
    ]


``filter_by_min_coder_doc_count``
*********************************

Filter to coders who have coded at least ``min_docs`` documents in the dataset:

.. code:: python

    "coder_filters": [
        ("filter_by_min_coder_doc_count", [], {"min_docs" 10})
    ]

``require_all_coders``
*********************************

Filter to documents that have been coded by all of the coders who participated in the sample:

.. code:: python

    "coder_filters": [
        ("require_all_coders", [], {})
    ]

``require_min_coder_count``
*********************************

Filter to documents that have been coded by at least N coders:

.. code:: python

    "coder_filters": [
        ("require_min_coder_count", [10], {})
    ]