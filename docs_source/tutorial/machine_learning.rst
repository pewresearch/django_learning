Machine learning
==================

So, we've got coding data for a sample - now for the fun part. Let's see if we can train a classifier on that
sample. At the end of the :doc:`Computing IRR tutorial </tutorial/computing_irr>` we pulled a dataset of our
in-house codes and compared that against Mechanical Turkers on the same sample. Now let's assume we've pulled a
much larger sample of documents called 'movie_review_sample_random_big" and had Turkers code that the same way.
Based on the original sample where we compared against our own in-house coding, we've determined that a
50% threshold works well for the "review_sentiment" variable. We can then pull a dataset of the collapsed Turk codes
for our bigger sample like so:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors
    from django_learning.models import Question

    base_class_id = (
        Question.objects\
            .filter(project__name="movie_reviews")
            .get(name="review_sentiment")
            .labels.get(value="0").pk
    )

    mturk = dataset_extractors["document_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random_big"],
        "question_names": ["review_sentiment"],
        "coder_aggregation_function": "mean",
        "coder_filters": [("exclude_experts", [], {})],
        "convert_to_discrete": True,
        "threshold": 0.5,
        "base_class_id": base_class_id
    }).extract(refresh=True)

(If we did our sampling in batches instead of one big sample, it's easy to include multiple samples in the
``sample_names`` parameter above.)

Defining a machine learning pipeline
-------------------------------------

Let's plug this dataset into a :doc:`machine learning pipeline </utils/machine_learning/pipelines>`:

.. code:: python

    def get_pipeline():

        from django_learning.utils.feature_extractors import feature_extractors
        from django_pewtils import get_model
        from sklearn.impute import SimpleImputer

        base_class_id = (
            get_model("Question", app_name="django_learning")
                .objects.filter(project__name="my_project")
                .get(name="my_question")
                .labels.get(value="0").pk
        )

        return {
            "dataset_extractor": {
                "name": "document_dataset",
                "parameters": {
                    "project_name": "movie_reviews",
                    "sample_names": ["movie_review_sample_random_big"],
                    "question_names": ["review_sentiment"],
                    "coder_aggregation_function": "mean",
                    "coder_filters": [("exclude_experts", [], {})],
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
                    ("tfidf", feature_extractors["tfidf"]()),
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="median"))
                ],
                "params": {
                    "tfidf": {
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
                },
            },
        }

Here, we're going to try something simple - we'll use the exact same parameters from above for the
``dataset_extractor``, and we'll try out an XGBoost model using basic TF-IDF features. We'll use 5-fold
cross-validation, and a random 20% holdout sample for testing. Alternatively, we could set the holdout to zero
(``"test_percent": 0``), and instead use the smaller sample that we coded in-house to evaluate the model:

.. code:: python

    def get_pipeline():

        from django_learning.utils.feature_extractors import feature_extractors
        from django_pewtils import get_model
        from sklearn.impute import SimpleImputer

        base_class_id = (
            get_model("Question", app_name="django_learning")
                .objects.filter(project__name="my_project")
                .get(name="my_question")
                .labels.get(value="0").pk
        )

        return {
            "dataset_extractor": {
                "name": "document_dataset",
                "parameters": {
                    "project_name": "movie_reviews",
                    "sample_names": ["movie_review_sample_random_big"],
                    "question_names": ["review_sentiment"],
                    "coder_aggregation_function": "mean",
                    "coder_filters": [("exclude_experts", [], {})],
                    "convert_to_discrete": True,
                    "threshold": 0.5,
                    "base_class_id": base_class_id
                },
                "outcome_column": "label_id",
            },
            "test_dataset_extractor": {
                "project_name": "movie_reviews",
                "sample_names": ["movie_review_sample_random"],
                "question_names": ["review_sentiment"],
                "coder_aggregation_function": "mean",
                "coder_filters": [("exclude_mturk", [], {})],
                "convert_to_discrete": True,
                "exclude_consensus_ignore": True
            },
            "model": {
                "name": "classification_xgboost",
                "cv": 5,
                "params": {},
                "fit_params": {"eval_metric": "error"},
                "use_sample_weights": True,
                "use_class_weights": True,
                "test_percent": 0.0,
                "scoring_function": "maxmin",
            },
            "pipeline": {
                "steps": [
                    ("tfidf", feature_extractors["tfidf"]()),
                    ("imputer", SimpleImputer(missing_values=np.nan, strategy="median"))
                ],
                "params": {
                    "tfidf": {
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
                },
            },
        }

We'll put this pipeline in a file called ``review_sentiment.py``.

Training a classifier
----------------------

And now we're ready for training!  Conveniently, there's
a built-in command for doing this.  We'll call our classifier "review_sentiment_classifier".

.. code:: bash

    python manage.py run_command django_learning_models_train_document_classifier review_sentiment_classifier review_sentiment

This will create a ``DocumentClassificationModel`` (see :doc:`Classification models </models/classification>` for more)
in the database, which we can then use to make predictions. Let's see how it did:

.. code:: bash

    from django_learning.models import DocumentClassificationModel

    model = DocumentClassificationModel.objects.get(name="review_sentiment_classifier")

    fold_scores = model.get_cv_prediction_results()
    >>> fold_scores[['outcome_column', 'precision', 'recall']]

    outcome_column  precision    recall
          label_id   0.937436  0.925000
      label_id__10   0.753333  0.683333
      label_id__11   0.958744  0.957882

    test_scores = model.get_test_prediction_results()
    >>> test_scores[['outcome_column', 'precision', 'recall']]

    outcome_column  precision    recall
          label_id   0.886029  0.875000
      label_id__10   0.500000  0.600000
      label_id__11   0.941176  0.914286


Looks like our model did okay. During our cross-validation, it looks like it averaged .75 precision and .68 recall
for our positive class (label_id__10) across the 5 folds. It didn't do quite as well on our test dataset:
.5 precision and .6 recall. Wonder if we can do better...

Optimizing the probability threshold
-------------------------------------

If we're working with a binary classifier - that is, there are only two values for your question and one of them
is a "positive" label and the other is "negative" - we can additionally tune our model by identifying an optimal
probability threshold. The ``DocumentClassificationModel.find_probability_threshold`` function loops over the
range (0, 1) and calculates the precision and recall scores across both the test set and each one of the folds in
your cross-validation folds. It then identifies the lowest precision or recall score across all of these datasets,
and picks the threshold that maximizes this minimum. In effect, you're identifying the threshold that makes the model's
worst performance as good as it can be. The result can often be a model with more balanced performance:

.. code:: bash

    model.find_probability_threshold(save=True)
    >>> model.probability_threshold
    0.85

    fold_scores = model.get_cv_prediction_results()
    >>> fold_scores[['outcome_column', 'precision', 'recall']]

    outcome_column  precision    recall
          label_id   0.964591  0.962500
      label_id__10   1.000000  0.683333
      label_id__11   0.959770  1.000000

    test_scores = model.get_test_prediction_results()
    >>> test_scores[['outcome_column', 'precision', 'recall']]

    outcome_column  precision  recall
          label_id   0.952703    0.95
      label_id__10   1.000000    0.60
      label_id__11   0.945946    1.00

Looks like recall for our positive class stayed the same - the model's not picking up any more true positives than
before - but the higher threshold is now reducing the number of false positives it's flagging. We're now getting
perfect precision across our five folds AND our test dataset!

Applying the classifier manually
---------------------------------

There are a number of ways to use a model once you've trained it. The most flexible is through the use of
the ``produce_prediction_dataset`` function, which can apply the model to arbitrary dataframes that are in
the same format as what was used to train the model. Since our pipeline only made use of the ``text`` column,
we could apply our model to our sampling frame like so:

.. code:: python

    import pandas as pd
    from django_learning.models import SamplingFrame

    frame = SamplingFrame.objects.get(name="movie_reviews")
    docs = pd.DataFrame.from_records(frame.documents.values("text"))
    df = model.produce_prediction_dataset(docs)
    >>> df
                                                 text label_id  probability
    plot : two teen couples go to a church party ,...       11     0.995819
    the happy bastard's quick movie review \ndamn ...       11     0.979824
    it is movies like these that make a jaded movi...       11     0.985981
     " quest for camelot " is warner bros . ' firs...       11     0.985981
    synopsis : a mentally unstable man undergoing ...       11     0.997317


We can also skip the step of having to compile a dataset ourselves by using a shortcut function that's unique to
``DocumentClassificationModels``:

.. code:: python

    df = model.apply_model_to_documents(frame.documents.all(), save=False)
    >>> df

    document_id                                               text       date document_type label_id  probability
              0  plot : two teen couples go to a church party ,... 2000-01-01  movie_review       11     0.995819
              1  the happy bastard's quick movie review \ndamn ... 2000-01-02  movie_review       11     0.979824
              2  it is movies like these that make a jaded movi... 2000-01-03  movie_review       11     0.985981
              3   " quest for camelot " is warner bros . ' firs... 2000-01-04  movie_review       11     0.985981
              4  synopsis : a mentally unstable man undergoing ... 2000-01-05  movie_review       11     0.997317

Or we can make it even simpler by letting the model auto-detect the sampling frame based on the samples it was
trained on:

.. code:: python

    df = model.apply_model_to_frame(save=False)
    >>> df

    document_id                                               text       date document_type label_id  probability
              0  plot : two teen couples go to a church party ,... 2000-01-01  movie_review       11     0.995819
              1  the happy bastard's quick movie review \ndamn ... 2000-01-02  movie_review       11     0.979824
              2  it is movies like these that make a jaded movi... 2000-01-03  movie_review       11     0.985981
              3   " quest for camelot " is warner bros . ' firs... 2000-01-04  movie_review       11     0.985981
              4  synopsis : a mentally unstable man undergoing ... 2000-01-05  movie_review       11     0.997317


Applying the classifier to the database
-----------------------------------------------

You'll notice, though, that the two functions above take a ``save`` parameter. Since sampling frames can be quite
large, it can take a long time to apply a model to your whole sampling frame - certainly, this isn't something that
we want to do over and over again. To that end, Django Learning stores model predictions in the database. In the same
way that coders create ``Code`` objects that correspond to a label given to a given document, classifiers in Django
Learning create ``Classification`` objects that do the same. We can easily apply the model to our whole sampling
frame and save the results by toggling ``save=True``:

.. code:: python

    model.apply_model_to_frame(save=True)

Instead of returning a datafame, the results will be saved to the database instead. There's also a handy built-in
command to do this instead of having to write out any code:

.. code:: bash

    python manage.py run_command django_learning_models_apply_document_classifier review_sentiment_classifier

After applying the model to the database, we can then access the classifications however we like:

.. code:: python

    df = pd.DataFrame.from_records(
        model.classifications.values(
            "document_id",
            "label__question__name",
            "label__value"
        )
    )
    >>> df

    document_id label__question__name label__value
             65         test_checkbox            1
            249         test_checkbox            0
            248         test_checkbox            0
            247         test_checkbox            0
            246         test_checkbox            0


    df = pd.DataFrame.from_records(
        frame.documents.values(
            "pk",
            "classifications__label___question__name",
            "classifications__label__value"
        )
    )
    >>> df

    pk classifications__label__question__name classifications__label__value
     0                          test_checkbox                             0
     1                          test_checkbox                             0
     2                          test_checkbox                             0
     3                          test_checkbox                             0
     4                          test_checkbox                             0

