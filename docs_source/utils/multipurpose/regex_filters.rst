Regex filters
--------------

Regex filters are used in several different places in Django Learning. They consist of very simple files that
return a compiled regular expression. For example:

.. code:: python

    import re

    def get_regex():
        return re.compile(r"cats")

Simply place this in a file in one of your ``settings.DJANGO_LEARNING_REGEX_FILTERS`` folders.


Usage in preprocessing
========================

The ``clean_text`` preprocessor accepts a ``regex_filters`` parameter that can be passed a list of regex filter names.
It uses the filters to filter documents to sentences that match to the regex. For example, if you had a regex filter
that matched to a specific set of names, you could use this filter to reduce documents to sentences that mention those
names.

.. code:: python

    from django_learning.utils.preprocessors import preprocessors

    text = "This sentence mentions cats. This one doesn't."
    preprocessor = preprocessors["clean_text"](regex_filters=["my_regex_filter"])
    >>> preprocessor.run(text)

    'sentence mention cat'

Usage in sampling
===================

Regex filters can be plugged in to custom sampling methods, indicating the proportion of a sample that
should match to the filter's regular expression:

.. code:: python

    def get_method():

        return {
            "sampling_strategy": "random",
            "stratify_by": None,
            "sampling_searches": [{"regex_filter": "my_regex_filter", "proportion": 0.5}],
        }


Usage in feature extractors
===========================

You can also use regex filters in conjunction with the ``regex_counts`` feature extractor (see Feature Extractors for more):

.. code:: python

    extractor = feature_extractors["regex_counts"](regex_filter="cats")
    >>> extractor.fit_transform(df)

        cats_count  cats_has_match  cats_count_sq  cats_count_log
    0          0.0               0            0.0             0.0
    1          0.0               0            0.0             0.0
    2          0.0               0            0.0             0.0
    3          0.0               0            0.0             0.0
    4          0.0               0            0.0             0.0