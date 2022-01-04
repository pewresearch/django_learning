Feature extractors
===================

For machine learning, Django Learning makes use of scikit-learn pipelines and grid searches. Part of these pipelines
are functions that extract features from your dataset that are then fed to the model. Django Learning provides a
template for developing feature extractors, along with a bunch of useful built-in ones.

Feature extractors extend the ``sklearn.base.BaseEstimator`` and ``sklearn.base.TransformerMixin``, so they
use the typical ``fit`` and ``transform`` paradigm common in sklearn. Feature extractors are passed
your dataset, and are expected to return a dataset with the same index, in a format that's usable by sklearn (Pandas
dataframes, sparse matrices, etc.)

All Django Learning feature extractors inherit from ``django_learning.utils.feature_extractors.BasicExtractor``. This
class provides a few additional features above and beyond what's built in to scikit-learn.

Row-level caching
-----------------

Feature extractors can be passed a ``cache_identifier`` name that initializes a local cache for row-level caching.
This happens behind the scenes and is toggled by LearningModel instances, but it means that when features are being
extracted during a grid search, the extractor will only process each unique row in your dataset once. If it encounters
the row again, it'll just load from the cache rather than recompute the features. Feature extractor parameters can
themselves be included in a grid search; caching is unique to each combination of parameters.

Not all feature extractors use row caching. If you wish to enable it in your own custom extractors, you can use the
``self.set_row_cache`` and ``self.get_row_cache`` functions.

Preprocessors
--------------

Feature extractors can also be passed a list of ``preprocessors``, another type of Django Learning utility that runs
on the data prior to extracting features. This makes it easy to plug in things like text cleaning. To pass in
preprocessors, you simply pass a tuple containing the name of the preprocessor and the parameters to use:

.. code:: python

    preprocessors = [
        ("clean_text", {"process_method": "lemmatize"})
    ]

Built-in feature extractors
---------------------------

In the examples below, ``df`` is assumed to be a dataframe that's been returned by a dataset extractor. For example:

.. code:: python

    df = dataset_extractors["raw_document_dataset"](sampling_frame_name="all_documents").extract()

Django field lookups
**********************************

This extractor fetches variables from the database. This can be useful if your Document objects have relations to
other models with important metadata. You simply pass this extractor a list of ``fields`` that are formatted the same
as you would use in a Django ``.values()`` query, and this extractor will build a dataframe of those values for you.

This extractor expects the dataset to have a ``document_id`` field, so it's best used in conjunction with a
``document_dataset`` dataset extractor.

.. code:: python

    from django_learning.utils.feature_extractors import feature_extractors

    extractor = feature_extractors["django_field_lookups"](
        fields=["facebook_post__likes"]
    )

    >>> extractor.fit_transform(df)

                    likes
    0               101.0
    1               13.0
    2               2.0
    3               498.0
    4               70.0


Ngram sets
*****************

Ngram sets are lists of words that belong to different categories, like sentiment dictionaries. You can apply them to
a dataset by adding the ``ngram_set`` feature extractor to a machine learning pipeline. The extractor will search for
and count up the words included in each category in the ngram set, and return a dataframe of categories in the
dictionary. This extractor expects a ``dictionary`` name and a ``feature_name_prefix``. It also expects text to be found
in a ``text`` column in your dataset.

.. code:: python

    from django_learning.utils.feature_extractors import feature_extractors

    extractor = feature_extractors["ngram_set"](
        dictionary="nrc_emotions",
        ngramset_name=None,
        feature_name_prefix="nrc",
        include_ngrams=False,
    )
    >>> extractor.fit_transform(df)

        nrc__anger  nrc__anticipation  nrc__disgust  nrc__fear  nrc__joy  nrc__negative  nrc__positive  nrc__sadness  nrc__surprise  nrc__trust
    0            1                  0             0          2         0              3              0             2              0           0
    1            2                  1             0          2         1              3              4             1              1           4
    2            2                  1             2          1         1              2              1             1              1           1
    3            0                  0             0          0         0              0              1             0              0           1
    4            0                  1             0          1         1              1              1             0              0           1


You can also apply a specific category by specifying an ``ngramset_name``:

.. code:: python

    extractor = feature_extractors["ngram_set"](
        dictionary="nrc_emotions",
        ngramset_name="anger",
        feature_name_prefix="nrc",
        include_ngrams=False,
    )
    >>> extractor.fit_transform(df)

        nrc__anger
    0            1
    1            2
    2            2
    3            0
    4            0


And in addition to aggregating, you can add columns for each ngram in the dictionary by specifying ``include_ngrams=True``:

.. code:: python

    extractor = feature_extractors["ngram_set"](
        dictionary="nrc_emotions",
        ngramset_name="anger",
        feature_name_prefix="nrc",
        include_ngrams=True,
    )
    >>> extractor.fit_transform(df)

        nrc__anger  nrc__ngram__abandoned  nrc__ngram__abandonment  nrc__ngram__abhor  nrc__ngram__abhorrent  ...  nrc__ngram__wrongful  nrc__ngram__wrongly  nrc__ngram__yell  nrc__ngram__yelp  nrc__ngram__youth
    0            1                      0                        0                  0                      0  ...                     0                    0                 0                 0                  0
    1            2                      0                        0                  0                      0  ...                     0                    0                 0                 0                  0
    2            2                      0                        0                  0                      0  ...                     0                    0                 0                 0                  0
    3            0                      0                        0                  0                      0  ...                     0                    0                 0                 0                  0
    4            0                      0                        0                  0                      0  ...                     0                    0                 0                 0                  0


Preprocessor
*****************

This feature extractor simply applies a list of preprocessors to your data. Normally you would pass them to one of the
more specific feature extractors below, but it can be useful for chaining things together in a pipeline. It gets run
on the ``text`` column of the dataset, so that needs to exist. The only parameter it needs is a list of ``preprocessors``.

.. code:: python

    results = feature_extractors["preprocessor"](
        preprocessors=[
            ("run_function", {"function": lambda x: x if "comedy" in x else ""})
        ]
    ).fit_transform(df)

    >>> results[results['text']!=""]

        document_id  sampling_weight label_id                                               text       date document_type
    13          254              1.0       23  even the best comic actor is at the mercy of h... 2000-09-11  movie_review
    28          585              1.0       23  the ads make " hanging up " seem like an upbea... 2001-08-08  movie_review
    63         1395              1.0       23  jake kasdan , son of one of the best screenwri... 2003-10-27  movie_review
    66         1473              1.0       23  a sci fi/comedy starring jack nicholson , pier... 2004-01-13  movie_review
    72         1532              1.0       23  i'm not quite sure what to say about mars atta... 2004-03-12  movie_review
    73         1555              1.0       23  harmless , silly and fun comedy about dim-witt... 2004-04-04  movie_review
    82         1664              1.0       23  if you've ever perused my college comedy diary... 2004-07-22  movie_review


Punctuation indicators
**********************************

Extracts a few different punctuation indicators and regex patterns from the ``text`` column - indicators of monetary
amounts and explanation points. This was useful for training a classifier to identify when politicians were talking
about constituent benefits. This could be a good template for building a custom feature extractor in the future.

.. code:: python

    extractor = feature_extractors["punctuation_indicators"](
        feature_name_prefix="punct"
    )
    >>> extractor.fit_transform(df)

        punct__dollars_count  punct__dollars_any  punct__dollars_alt_count  punct__dollars_alt_any  punct__amounts_count  punct__amounts_any  punct__exclamation_count  punct__exclamation_any
    0                      0                   0                         0                       0                     0                   0                         0                       0
    1                      0                   0                         0                       0                     0                   0                         0                       0
    2                      0                   0                         0                       0                     0                   0                         0                       0
    3                      0                   0                         0                       0                     0                   0                         0                       0
    4                      0                   0                         0                       0                     0                   0                         0                       0

Regex counts
*****************

A more sophisticated and flexible option is to make use of Django Learning regex filters, which can also be used to
extract features using this feature extractor. By providing the name of a ``regex_filter``, the extractor will
produce a binary flag indicating documents that match to the regex (``_has_match``) as well as the number of
matches (``_count``) and the squared and logged version of the counts. Requires a ``text`` column.

.. code:: python

    extractor = feature_extractors["regex_counts"](regex_filter="cats")
    >>> extractor.fit_transform(df)

        cats_count  cats_has_match  cats_count_sq  cats_count_log
    0          0.0               0            0.0             0.0
    1          0.0               0            0.0             0.0
    2          0.0               0            0.0             0.0
    3          0.0               0            0.0             0.0
    4          0.0               0            0.0             0.0


TF-IDF
*****************

The bread and butter of traditional NLP - this extractor converts the ``text`` column into a TF-IDF matrix using
the sklearn ``TfidfVectorizer``. Keyword arguments get forwarded to sklearn, but this extractor also provides the
ability to run preprocessors on the text - and you also get

.. code:: python

    extractor = feature_extractors["tfidf"](
        max_df=0.9,
        min_df=2,
        ngram_range=(1, 1),
        preprocessors=[
            (
                "clean_text",
                {"process_method": "lemmatize", "stopword_sets": ["english"]},
            )
        ],
    )

    >>> extractor.fit_transform(df)

    <100x235 sparse matrix of type '<class 'numpy.float64'>'
        with 677 stored elements in Compressed Sparse Row format>


Google Word2Vec
*****************

If you download the ``GoogleNews-vectors-negative300.bin.gz`` pre-trained Word2Vec model and place it in the
``settings.FILE_ROOT`` folder, you can extract features using this model with the ``google_word2vec`` feature extractor.
No parameters are required, but you can pass a ``limit`` that will load the first N features from the model (it has
millions of features, but if you only want to load the first 100k, you could pass ``limit=100000``, and it will run
more efficiently, but less effectively).

.. code:: python

    extractor = feature_extractors["google_word2vec"](limit=None)


Topics
*****************

If you've trained a topic model, you can easily plug the model in and grab features using the ``topics`` feature
extractor. Simply pass the model name in:

.. code:: python

    extractor = feature_extractors["topics"](model_name="my_topic_model")



Making a custom feature extractor
----------------------------------

Custom feature extractors require ``fit``, ``transform``, and ``get_feature_names`` functions. Looking at the source
code for the built-in feature extractors is a good way to see how to build one of your own.