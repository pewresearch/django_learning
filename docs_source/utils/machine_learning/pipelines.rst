Pipelines
==========

Pipelines are how you stitch everything together for machine learning in Django Learning.
Pipeline files require a ``get_pipeline`` function that returns a dictionary with three keys:

Dataset extractor
-----------------

The first component of a Django Learning pipeline is the dataset extractor. This dictionary should contain the
name of a valid Django Learning dataset extractor (either your own, or more likely, one of the built-in ones),
the column in the dataset you want to predict (the ``outcome_column``), and parameters to be passed to the
dataset extractor. For example:

.. code:: python

    "dataset_extractor": {
        "name": "document_dataset",
        "parameters": {
            "project_name": "my_project",
            "sandbox": False,
            "sample_names": ["my_sample"],
            "question_names": ["my_question"],
            "document_filters": [],
            "coder_filters": [],
            "balancing_variables": [],
            "ignore_stratification_weights": False,
            "standardize_coders": False,
            "coder_aggregation_function": "mean",
            "convert_to_discrete": True,
            "threshold": 0.5,
            "base_class_id": base_class_id
        },
        "outcome_column": "label_id",
    }

Note the reference to the variable ``base_class_id`` above. The reason why we define pipelines in Python functions is
that we can do handy stuff like this, prior to compiling our pipeline dictionary:

.. code:: python

    from django_pewtils import get_model
    base_class_id = (
        get_model("Question", app_name="django_learning")
            .objects.filter(project__name="my_project")
            .get(name="my_question")
            .labels.get(value="0").pk
    )



Model
-----------------

The next component in a pipeline is the model. This specifies the algorithm to use, as well as various decisions about
weighting and scoring:

.. code:: python

    "model": {
        "name": "classification_xgboost",
        "cv": 5,
        "params": {},
        "fit_params": {"eval_metric": "error"},
        "use_sample_weights": True,
        "use_class_weights": True,
        "test_percent": 0.2,
        "scoring_function": "maxmin",
    }

Here's what all of this does:

    * ``cv``: run k-fold cross-validation (0 or 1 disables cross-validation, in this example we're doing 5-fold)
    * ``params``: parameters that get passed to the model on initialization
    * ``fit_params``: parameters that get passed to the model on ``fit``
    * ``use_sample_weights``: whether or not to use sampling weights during training
    * ``use_class_weights``: whether or not to use class-balancing weights during training (based on the distribution of your outcome column)
    * ``test_percent``: proportion of the rows in the dataset to be held out entirely (even during cross-validation) as a test dataset
    * ``scoring_function``: name of the Django Learning scoring function to use to pick the best model

Pipeline
-----------------

The final section of your pipeline - the ``pipeline`` configuration - specifies your preprocessing and feature
extraction. This mirrors how sklearn's grid search pipelines work.

First, you'll need a ``steps`` section. This should take the form of a list of steps through which your dataset will
be passed. Each step should be a tuple of a name and a valid sklearn processor or Django Learning feature extractor.

For example, we might wish to extract TF-IDF features, so we could add this to our pipeline:

.. code:: python
    "steps": [
        ("tfidf", feature_extractors["tfidf"]())
    ]

But we might also want to use that same extractor with different parameters - maybe binary flags for words instead
of their TF-IDF weights. Because we're going to be grid searching over parameters, we'll specify them in the next
section - but for now, we want to add two sets of TF-IDF features to our pipeline. We can do this using a
``sklearn.pipeline.FeatureUnion``, which itself takes a list of tuples and bundles together multiple steps as
something to do concurrently rather than sequentially. The output then gets horizontally concatenated together.

.. code:: python

    tfidf_features = FeatureUnion([
        ("tfidf_counts", feature_extractors["tfidf"]()),
        ("tfidf_bool", feature_extractors["tfidf"]())
    ])

    "steps": [
        ("features", tfidf_features)
    ]

This should do it for a simple pipeline. To top it off, we'll add an sklearn ``SimpleImputer`` to the pipeline to
fill in any missing values.

.. code:: python

    from sklearn.impute import SimpleImputer

    "steps": [
        ("features", tfidf_features),
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median"))
    ]

Next we need to specify the other half of our pipeline configuration. We have the steps, now we need to pick out
the parameters that will get passed to each of our extractors, and decide if we want to grid search anything. To do this,
we'll add a ``params`` section to the pipeline config. For each named extractor that requires parameters in our steps
above - ``tfidf_counts`` and ``tfidf_bool`` - we need to add those parameters to this section. However, unlike when
we pass parameters to these extractors directly, this time we're going to pass all of our parameters in as _lists_,
each one containing all of the different values we want to test in our grid search.

For our normal TF-IDF pipeline, we might try something like this:

.. code: python

    "tfidf_counts": {
        "max_df": [0.9],
        "min_df": [5, 10],
        "max_features": [None],
        "ngram_range": [[1, 4]],
        "use_idf": [True],
        "norm": ["l2"],
        "binary": [False],
        "sublinear_tf": [True],
        "preprocessors": [
            [("clean_text", {"process_method": "lemmatize", "stopword_sets": ["english"})]
            [("clean_text", {"process_method": "stem", "stopword_sets": ["english"})]
        ],
    }

This looks like a lot, but it's actually not. For most parameters, we're going to pass just one value. But for
``min_df``, we'll try out a minimum of 5 and 10 documents - maybe the extra granularity will help, or maybe it will
cause overfitting, we'll see!  We'll also try out two different ways of preprocessing our text - one with lemmatization,
and another with stemming.

For ``tfidf_bool``, we'll basically do the same thing, except pass parameters that create binary counts rather than
TF-IDF weights, by settting ``sublinear_tf=False``, ``norm=None``, ``use_idf=False``, and ``binary=True``.

Putting it all together
----------------------------------

Putting all of this together, and we get a pipeline file that looks like this:

.. code:: python

    def get_pipeline():

        from django_learning.utils.feature_extractors import feature_extractors
        from django_pewtils import get_model
        from sklearn.pipeline import FeatureUnion
        from sklearn.impute import SimpleImputer

        base_class_id = (
            get_model("Question", app_name="django_learning")
                .objects.filter(project__name="my_project")
                .get(name="my_question")
                .labels.get(value="0").pk
        )

        tfidf_features = FeatureUnion([
            ("tfidf_counts", feature_extractors["tfidf"]()),
            ("tfidf_bool", feature_extractors["tfidf"]())
        ])

        return {
            "dataset_extractor": {
                "name": "document_dataset",
                "parameters": {
                    "project_name": "my_project",
                    "sandbox": False,
                    "sample_names": ["my_sample"],
                    "question_names": ["my_question"],
                    "document_filters": [],
                    "coder_filters": [],
                    "balancing_variables": [],
                    "ignore_stratification_weights": False,
                    "standardize_coders": False,
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 0.5,
                    "base_class_id": base_class_id
                },
                "outcome_column": "label_id",
            },
            "model": {
                "name": "classification_xgboost",
                "cv": 5,
                "params": {},
                "fit_params": {"eval_metric": "error"},
                "use_sample_weights": True,
                "use_class_weights": True,
                "test_percent": 0.2,
                "scoring_function": "maxmin",
            },
            "pipeline": {
                "steps": [
                    ("features", tfidf_features),
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="median"))
                ],
                "params": {
                    "tfidf_counts": {
                        "max_df": [0.9],
                        "min_df": [5, 10],
                        "max_features": [None],
                        "ngram_range": [[1, 4]],
                        "use_idf": [True],
                        "norm": ["l2"],
                        "binary": [False],
                        "sublinear_tf": [True],
                        "preprocessors": [
                            [("clean_text", {"process_method": "lemmatize", "stopword_sets": ["english"})]
                            [("clean_text", {"process_method": "stem", "stopword_sets": ["english"})]
                        ],
                    },
                    "tfidf_bool": {
                        "max_df": [0.9],
                        "min_df": [5, 10],
                        "max_features": [None],
                        "ngram_range": [[1, 4]],
                        "use_idf": [False],
                        "norm": [None],
                        "binary": [True],
                        "sublinear_tf": [False],
                        "preprocessors": [
                            [("clean_text", {"process_method": "lemmatize", "stopword_sets": ["english"})]
                            [("clean_text", {"process_method": "stem", "stopword_sets": ["english"})]
                        ],
                    }
                },
            },
        }


Using separate test datasets
---------------------------------------------------

You can also specify a separate test dataset by adding dataset extractor configuration to an additional
``test_dataset_extractor`` key in your pipeline. By default, test datasets are held-out as a proportion of your primary
dataset using the ``test_percent`` parameter (see the Model section above), but if you set the test percent to zero and
add a ``test_dataset_extractor`` to your pipeline, Django Learning will use that instead.