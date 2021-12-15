Dataset extractors
====================

Django Learning uses something called "dataset extractors" to load and cache datasets of documents and codes.
All dataset extractors inherit from the ``django_learning.utils.dataset_extractors.DatasetExtractor`` class.
This class provides automatic caching based on the source code of the
dataset extractor itself, and you can expand this function in your own extractor
to add additional attributes to the hash key that make the dataset unique. The only other
requirement for a custom dataset extractor is a ``_get_dataset`` function that returns a
Pandas dataframe. Here's an example of a custom dataset extractor that fetches Django Learning
documents that contain a certain keyword. The first time you run the ``extract`` function, it'll generate and cache
the dataframe. The next time you run it, it'll load directly from the cache. To recompute the dataframe,
you can run ``extract(refresh=True)``:

.. code:: python

        class MyDatasetExtractor(DatasetExtractor):

            def __init__(self, **kwargs):
                self.keyword = kwargs.get("keyword", None)

            def get_hash(self, **kwargs):
                hash_key = super(Extractor, self).get_hash(**kwargs)
                hash_key += str(self.keyword)
                return self.cache.file_handler.get_key_hash(hash_key)

            def _get_dataset(self, **kwargs):
                return pd.DataFrame.from_records(
                    Document.objects.filter(text__contains=self.keyword).values()
                )

        >>> MyDatasetExtractor(keyword="cats").extract(refresh=True)

So that's a basic dataset extractor. Once defined, you can use it in a machine learning pipeline
simply by referencing the name. Of course, to be useful, you need to define an outcome column that
exists in the dataframe that you want to predict. Let's assume that the example above has such a
column; in this case, you can set up your pipeline (see Pipelines for more details) to use your
extractor like so:

.. code:: python

    "dataset_extractor": {
        "name": "my_dataset_extractor",  # name of the extractor file
        "parameters": {"keyword": "cats"},
        "outcome_column": "my_outcome_column",
    }

That's all you need to configure the pipeline to use your dataset extractor and run the
model on your outcome column. Since our example above will have a "text" column with document
text, we could then add built-in feature extractors like tfidf to our pipeline, and start
training and applying a model.

Built-in dataset extractors
---------------------------

In most cases, you won't need to create a custom dataset extractor. If you're using the Django
Learning coding interface, you'll already have documents and codes that you probably want to extract
and perhaps pass to a machine learning pipeline. There are three main extractors for grabbing
datasets of documents and codes at different levels of granularity:

Document-coder-label datasets
******************************

The most granular extractor returns a dataframe with one row per code, for every coder and every
document in one or more samples, for a given set of questions. The basic parameters are as follows:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    extractor = dataset_extractors["document_coder_label_dataset"](**{
        "project_name": "my_project_name",
        "sample_names": ["my_sample_name"],
        "question_names": ["my_question_name"],
        "document_filters": [],
        "coder_filters": [],
        "code_filters": [],
        "balancing_variables": [],
        "ignore_stratification_weights": False,
    })
    >>> extractor.extract(refresh=True)

    pk_x  coder_id coder_name  coder_is_mturk  document_id       date  label_id  ...                                               text  sampling_weight approx_weight strat_weight  keyword_weight  additional_weight document_type
     302         4     coder2           False            0 2000-01-01        23  ...  plot : two teen couples go to a church party ,...              1.0           1.0         None            None               None  movie_review
     301         3     coder1           False            0 2000-01-01        23  ...  plot : two teen couples go to a church party ,...              1.0           1.0         None            None               None  movie_review
     304         4     coder2           False            4 2000-01-05        23  ...  synopsis : a mentally unstable man undergoing ...              1.0           1.0         None            None               None  movie_review
     303         3     coder1           False            4 2000-01-05        23  ...  synopsis : a mentally unstable man undergoing ...              1.0           1.0         None            None               None  movie_review
     306         4     coder2           False            5 2000-01-06        23  ...  capsule : in 2176 on the planet mars police ta...              1.0           1.0         None            None               None  movie_review


The ``project_name`` keyword argument should refer to your coding project, etc. If you want to
do some testing on Mechanical Turk, you may have set up your project with ``sandbox=True``, in which
case pass this kwarg accordingly. Document, coder and code filters are explained in their
respective sections, as are balancing variables.

Stratification variables get defined when you specify your sampling methods (see Sampling methods)
via the ``stratify_by`` parameter, and by default, Django Learning's built-in extractors will combine
all of the variables that you stratified your sample(s) by, and blend those into the training weights that get
computed. You can disable this by setting ``ignore_stratification_weights=False``.

If you pass multiple questions to ``question_names``, you'll get a binary string representation of all of the
label options across the combined set of questions, like so:

.. code:: python

    extractor = dataset_extractors["document_coder_label_dataset"](**{
        "project_name": "my_project_name",
        "sample_names": ["my_sample_name"],
        "question_names": ["my_question_name", "another_question"],
        "document_filters": [],
        "coder_filters": [],
        "code_filters": [],
        "balancing_variables": [],
        "ignore_stratification_weights": False,
    })
    >>> extractor.extract(refresh=True)

    document_id  coder_id coder_name  coder_is_mturk  sampling_weight       date label_id                                               text document_type
              0         3     coder1           False              1.0 2000-01-01    00101  plot : two teen couples go to a church party ,...  movie_review
              0         4     coder2           False              1.0 2000-01-01    00001  plot : two teen couples go to a church party ,...  movie_review
              4         3     coder1           False              1.0 2000-01-05    00101  synopsis : a mentally unstable man undergoing ...  movie_review
              4         4     coder2           False              1.0 2000-01-05    00001  synopsis : a mentally unstable man undergoing ...  movie_review
              5         3     coder1           False              1.0 2000-01-06    00101  capsule : in 2176 on the planet mars police ta...  movie_review


Document-coder datasets
******************************

This extractor consolidates coding data at one level higher than the ``document_coder_label_dataset`` extractor.
It collapses the dataframe to the document-coder level. If you pass a single value to ``question_names``, each code
will be converted into its own binary column in the form ``label_[primary_key]``. If you pass multiple questions,
they'll be concatenated into a binary string representation and a column for each unique combination will be created
in the form of ``label_00101``, for example.

Document-coder dataset extractors accept an additional kwargs, ``standardize_coders``. If this is set to true,
all of the code columns will be standardized with z-scores for each coder. This can be useful if some coders
have a generally higher or lower propensity to pick certain codes.

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    # One question name:

    extractor = dataset_extractors["document_coder_dataset"](**{
        "project_name": "my_project_name",
        "sample_names": ["my_sample_name"],
        "question_names": ["my_question_name"],
        "document_filters": [],
        "coder_filters": [],
        "code_filters": [],
        "balancing_variables": [],
        "ignore_stratification_weights": False,
        "standardize_coders": False,
    })
    >>> extractor.extract(refresh=True)

    document_id  coder_id  label_10  label_11 coder_name  coder_is_mturk  sampling_weight       date                                               text document_type
              0         1         0         1     coder1           False              1.0 2000-01-01  plot : two teen couples go to a church party ,...  movie_review
              0         2         0         1     coder2           False              1.0 2000-01-01  plot : two teen couples go to a church party ,...  movie_review
              4         1         0         1     coder1           False              1.0 2000-01-05  synopsis : a mentally unstable man undergoing ...  movie_review
              4         2         0         1     coder2           False              1.0 2000-01-05  synopsis : a mentally unstable man undergoing ...  movie_review
              5         1         0         1     coder1           False              1.0 2000-01-06  capsule : in 2176 on the planet mars police ta...  movie_review

    # And if we pass two question names:

    extractor = dataset_extractors["document_coder_dataset"](**{
        "project_name": "my_project_name",
        "sample_names": ["my_sample_name"],
        "question_names": ["my_question_name", "another_question],
        "document_filters": [],
        "coder_filters": [],
        "code_filters": [],
        "balancing_variables": [],
        "ignore_stratification_weights": False,
        "standardize_coders": False,
    })
    >>> extractor.extract(refresh=True)

    document_id  coder_id  label_00001  label_00010  label_00101  label_00110  label_01001  ...  label_10010  coder_name coder_is_mturk  sampling_weight       date                                               text document_type
              0         1            0            0            1            0            0  ...            0      coder1          False              1.0 2000-01-01  plot : two teen couples go to a church party ,...  movie_review
              0         2            1            0            0            0            0  ...            0      coder2          False              1.0 2000-01-01  plot : two teen couples go to a church party ,...  movie_review
              4         1            0            0            1            0            0  ...            0      coder1          False              1.0 2000-01-05  synopsis : a mentally unstable man undergoing ...  movie_review
              4         2            1            0            0            0            0  ...            0      coder2          False              1.0 2000-01-05  synopsis : a mentally unstable man undergoing ...  movie_review
              5         1            0            0            1            0            0  ...            0      coder1          False              1.0 2000-01-06  capsule : in 2176 on the planet mars police ta...  movie_review


Document datasets
******************************

This extractor is what you will use most often, as it's what's typically used to train machine
learning pipelines.

The final aggregation level is to collapse everything down to a document-level dataset,
one row per document. If you have multiple coders, this requires some way of aggregating them
together. To this end, document dataset extractors require a few additional kwargs:

    ** ``coder_aggregation_function``: this specifies how to consolidate each code column; the
        options are "mean", "median", "max" and "min". The dataset is grouped by document and
        this function gets applied across all the coders for that document.
    ** ``convert_to_discrete``: if False (default), you'll be given a dataset with separate columns for
        each label value (e.g. ``label_[primary_key]``) collapsed by your ``coder_aggregation_function``.
        This can result in continuous variables across the range 0-1. If you set this to True, the
        extractor will instead convert these into discrete (i.e. categorical) values. This will result
        in a column for each variable in your ``question_names`` with values representing labels
        in the form ``label_[primary_key]``. To do this, you need to specify two additional parameters:
    ** ``base_class_id``: the label that will be selected by default if the aggregated value is below the threshold
        required to mark something as positive
    ** ``threshold``: a value (0-1) that defines when something should be marked as positive

For starters, let's just aggregate coders by averaging:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    extractor = dataset_extractors["document_dataset"](**{
        "project_name": "my_project_name",
        "sample_names": ["my_sample_name"],
        "question_names": ["my_question_name"],
        "document_filters": [],
        "coder_filters": [],
        "code_filters": [],
        "balancing_variables": [],
        "ignore_stratification_weights": False,
        "standardize_coders": False,
        "coder_aggregation_function": "mean",
        "convert_to_discrete": False,
        "threshold": None,
        "base_class_id": None
    })
    >>> extractor.extract(refresh=True)

    document_id  label_10  label_11  sampling_weight                                               text       date document_type
             34       0.0       1.0              1.0  if you're into watching near on two hours of b... 2000-02-04  movie_review
            128       0.0       1.0              1.0  susan granger's review of " ghosts of mars " (... 2000-05-08  movie_review
             35       0.0       1.0              1.0  sean connery stars as a harvard law professor ... 2000-02-05  movie_review
             40       0.5       0.5              1.0  lengthy and lousy are two words to describe th... 2000-02-10  movie_review
             67       0.0       1.0              1.0   " first rule of fight club is , don't talk ab... 2000-03-08  movie_review

Here we can see that some of the labels are marked as 0.5, indicating the proportion of the two coders who
labeled the document as such. If we pass two question names in, we get something similar, except with a
column for each unique code combination:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    extractor = dataset_extractors["document_dataset"](**{
        "project_name": "my_project_name",
        "sample_names": ["my_sample_name"],
        "question_names": ["my_question_name", "another_question"],
        "document_filters": [],
        "coder_filters": [],
        "code_filters": [],
        "balancing_variables": [],
        "ignore_stratification_weights": False,
        "standardize_coders": False,
        "coder_aggregation_function": "mean",
        "convert_to_discrete": False,
        "threshold": None,
        "base_class_id": None
    })
    >>> extractor.extract(refresh=True)

    document_id  label_00001  label_00010  label_00101  label_00110  label_01001  label_10001  label_10010  sampling_weight                                               text       date document_type
              0          0.5          0.0          0.5          0.0          0.0          0.0          0.0              1.0  plot : two teen couples go to a church party ,... 2000-01-01  movie_review
              4          0.5          0.0          0.5          0.0          0.0          0.0          0.0              1.0  synopsis : a mentally unstable man undergoing ... 2000-01-05  movie_review
              5          0.5          0.0          0.5          0.0          0.0          0.0          0.0              1.0  capsule : in 2176 on the planet mars police ta... 2000-01-06  movie_review
              7          0.5          0.0          0.0          0.0          0.0          0.5          0.0              1.0  that's exactly how long the movie felt to me .... 2000-01-08  movie_review
              8          0.5          0.0          0.5          0.0          0.0          0.0          0.0              1.0  call it a road trip for the walking wounded . ... 2000-01-09  movie_review

If we want to collapse our data into discrete (categorical) variables so we can train a classifier,
we can pass ``convert_to_discrete=True``. Depending on the values you pass for ``threshold`` and
``base_class_id``, the behavior will be different:

    ** **Threshold and base class are both set:** if the value in a particular column is above the threshold, that code
        value will be returned, otherwise it will be set to the ``base_class_id`` value
    ** **Threshold is set, base class is not:** returns the column with the maximum value, but cases that fall below
        the threshold will be NoneTyoe (e.g. if you're using ``mean`` to aggregate the coders and the threshold is
        set to 0.8 and there were 5 coders, cases where 4 of the 5 coders picked a code will be returned as such,
        but cases where there was a 2-3 split will be NoneType)
    ** **Base class is set, threshold is not:** returns the column with the maximum value, or the base class if all
        other values are zero. If there's a tie between two columns, the selection may be arbitrary.
    ** **Neither threshold nor base class are set**: returns the column with the maximum value. If there's a tie
        between two columns, the selection may be arbitrary.

Typically you should set both the ``threshold`` and ``base_class_id``. Here's an example of an extractor that
averages across coders and converts the codes into an aggregated discrete value for each document. In this example,
our code question has three options (10, 11 and 12 are the primary keys), and we set the ``base_class_id`` to 10.
Setting the threshold to 0.4, we'll mark a case as positive if at least half of the coders marked it as such.
If multiple codes are above this threshold (40% picked one label and 60% picked another), we'll get the highest of
the two. If no codes were above that threshold (say, a third of the coders picked each label), then we'll mark the
document with the base class.

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    extractor = dataset_extractors["document_dataset"](**{
        "project_name": "my_project_name",
        "sample_names": ["my_sample_name"],
        "question_names": ["my_question_name", "another_question"],
        "document_filters": [],
        "coder_filters": [],
        "code_filters": [],
        "balancing_variables": [],
        "ignore_stratification_weights": False,
        "standardize_coders": False,
        "coder_aggregation_function": "mean",
        "convert_to_discrete": True,
        "threshold": 0.4,
        "base_class_id": 10
    })
    >>> extractor.extract(refresh=True)

    document_id  sampling_weight label_id                                               text       date document_type
           134              1.0       10  what makes reindeer games even more disappoint... 2000-05-14  movie_review
           113              1.0       12  one of the contributors to the destruction of ... 2000-04-23  movie_review
           122              1.0       11  when respecting a director , you must also res... 2000-05-02  movie_review
            64              1.0       10  rated : r for strong violence , language , dru... 2000-03-05  movie_review
           138              1.0       10  my opinion on a film can be easily swayed by t... 2000-05-18  movie_review

Raw document datasets
******************************

The raw document dataset extractor does exactly what it sounds like - it just pulls a dataframe of documents,
no coding data attached. You can pass it either a list of Document primary keys via the ``document_ids`` parameter,
or the name of a sampling frame via ``sampling_frame_name``. It also accepts ``document_filters``, which functions
the same as with the other built-in extractors above.

..code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    extractor = dataset_extractors["raw_document_dataset"](
        document_ids=[],
        sampling_frame_name="my_sampling_frame",
        document_filters=[]
    )
    >>> extractor.extract(refresh=True)

    document_id                                               text       date document_type
              0  plot : two teen couples go to a church party ,... 2000-01-01  movie_review
              1  the happy bastard's quick movie review \ndamn ... 2000-01-02  movie_review
              2  it is movies like these that make a jaded movi... 2000-01-03  movie_review
              3   " quest for camelot " is warner bros . ' firs... 2000-01-04  movie_review
              4  synopsis : a mentally unstable man undergoing ... 2000-01-05  movie_review


Model prediction datasets
******************************

If you've trained a machine learning model to make predictions, you can apply it to another dataset using this extractor,
as long as that dataset has all of the columns necessary for the model's pipeline to function. This extractor will
return the raw predictions (and, if applicable, probabilities) from the sklearn model.

.. code:: python

    # Grab a dataset of documents
    df = dataset_extractors["raw_document_dataset"](
        document_ids=[], sampling_frame_name="all_documents", document_filters=[]
    ).extract()

    # Grab a model you've trained
    model = DocumentClassificationModel.objects.get(name="test")

    # Make predictions on the dataset using the model
    df = dataset_extractors["model_prediction_dataset"](
        dataset=df, learning_model=model, disable_probability_threshold_warning=True
    ).extract()

    >>> df

    document_id                                               text       date document_type label_id  probability
              0  plot : two teen couples go to a church party ,... 2000-01-01  movie_review       23     0.898995
              1  the happy bastard's quick movie review \ndamn ... 2000-01-02  movie_review       23     0.985692
              2  it is movies like these that make a jaded movi... 2000-01-03  movie_review       23     0.956804
              3   " quest for camelot " is warner bros . ' firs... 2000-01-04  movie_review       23     0.988936
              4  play it to the bone is a punch-drunk mess of a... 2000-05-27  movie_review       22     0.858744


If you've optimized the probability threshold, it won't be applied, and by default it will warn you of this.
The preferred way of making predictions with a trained model is to use the model directly via one of its functions,
like ``produce_prediction_dataset`:

.. code:: python

    # Grab a dataset of documents
    df = dataset_extractors["raw_document_dataset"](
        document_ids=[], sampling_frame_name="all_documents", document_filters=[]
    ).extract()

    # Grab a model you've trained
    model = DocumentClassificationModel.objects.get(name="test")

    # Apply the model to the dataset
    df = model.produce_prediction_dataset(df)