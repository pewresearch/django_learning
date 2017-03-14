Balancing variables
====================

When extracting a dataset, you can use categorical variables to balance the weights in your dataset
across different groups. This can be useful during machine learning, if you want to make sure that
different groups of documents are given equal weight during training.

When extracting a `document_dataset`, `document_coder_datset`, or `document_coder_label` dataset, you can
pass in a keyword argument containing the names of `balancing_variables` that correspond to functions defined
in your `settings.DJANGO_LEARNING_BALANCING_VARIABLES` folders. Weights will be computed to evenly balance all
classes within the partitions (using the combinations if multiple variables are included) and the results will
be contained in a `balancing_weight` column in the returned dataset. If you specify a balancing weight in the
training dataset for a machine learning pipeline, the `LearningModel` will use the balancing weight from the
dataset when computing its training weight. If class weights and/or sampling weights are included, these will
be multiplied by the balancing weight. Balancing weights will NOT be used when evaluating model performance.
They exist in Django Learning solely for the training phase of machine learning, or simply for your own
convenience when you're extracting datasets.

Defining balancing weights
--------------------------

Balancing weight files require a ``var_mapper`` function that is passed a row in the dataset,
and returns a value for that row. The values should correspond to the categories of your documents
that you wish to balance. Here's an example of a balancing weight function that would balance
documents by the month they were produced:

.. code:: python

    from django_pewtils import get_model

    def var_mapper(x):

        doc = get_model("Document", app_name="django_learning").objects.get(
            pk=x["document_id"]
        )
        if doc.date and doc.date.month and doc.date.year:
            return "{}_{}".format(doc.date.year, doc.date.month)
        else:
            return None


Using balancing weights
-----------------------

You can access balancing weight functions directly like so:

.. code:: python

    from django_learning.utils.balancing_variables import balancing_variables
    balancing_variables.keys()

However, in practice, you shouldn't ever need to do so. Instead, you can specify balancing
weights as a key in your dataset extractor parameters, either directly when working with
dataset extractors:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors
    params = {
        "project_name": "my_project",
        "sample_names": ["my_sample"],
        "question_names": ["my_question"],
        "balancing_variables": ["my_balancing_variable"]
    }
    extractor = dataset_extractors["document_dataset"](**params)
    df = extractor.extract(refresh=True)

Or as part of a machine learning pipeline:

.. code:: python

    {
        "dataset_extractor": {
            "name": "document_dataset",
            "parameters": {
                "project_name": "my_project",
                "sample_names": ["my_sample"],
                "question_names": ["my_question"],
                "balancing_variables": ["my_balancing_variable"]  # goes here
            },
            "outcome_column": "label_id",
        },
        "model": { ... }
    }


Built-in dataset coder filters
-------------------------------

``document_type``
******************

Django Learning provides a built-in balancing variable called "document_type", which weights documents
evenly based on their OneToOne relations with different models in your app. If the Document objects in
your sample belong to either a FacebookPost or Tweet model in your app, for example, using this balancing variable
will make sure that FacebookPost Documents and Tweet Documents are weighted equally such that the
sum of the weights in each group comprise 50% of the total.