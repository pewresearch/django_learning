Sampling methods
=================

Sampling methods specify a way to pull a sample of documents from a frame. They are defined in sampling method config
files which require a ``get_method`` function that returns a dictionary with the following parameters:

.. code:: python

    def get_method():

        return {
            "sampling_strategy": "random",
            "stratify_by": None,
            "sampling_searches": [],
            "additional_weights": {}
        }

Parameters
----------

sampling_strategy
*****************

Gets passed to ``pewanalytics.sampling.SampleExtractor``. Valid options are:

    - "all": selects all documents
    - "random": Random sample
    - "stratify": Proportional stratification, method from Kish, Leslie. "Survey sampling." (1965). Chapter 4.
    - "stratify_even": Sample evenly from each strata (will obviously not be representative)
    - "stratify_guaranteed": Proportional stratification, but the sample is guaranteed to contain at least one \
        observation from each strata (if sample size is small and/or there are many small strata, the resulting \
        sample may be far from representative)

stratify_by
************

A Django field lookup (from the Document table) that selects a variable to stratify on (e.g. "date").

.. code:: python

    "stratify_by": "politician__party"

sampling_searches
*****************

Used for keyword oversampling. You can put a list of regex filters along with their desired proportions, and
Django Learning will attempt to pull a sample of documents that match to the regexes in the requested proportions.
Since regex filters may overlap, the proportions won't be exact. Weights will be computed using the combination of
all of the different filters, so you shouldn't use multiple rare filters otherwise you may wind up with "weight
explosion". The proportions should either sum to 1.0 or less than that - the remainder will be filled with documents
that do not match to any of the regex filters.

.. code:: python

    "regex_filters": [
        {"regex_filter": "cats", "proportion": 0.1},
        {"regex_filter": "dogs", "proportion": 0.1}
    ]

additional_weights
*******************

You can add additional weighting variables using Django field lookups and mapper functions. These values won't be
used in sampling, but they'll be used in weighting to ensure that your sample remains representative across all
values of the additional weighting variables. Each additional weight should be given a name, which should map to a
dictionary with a Django field lookup, and a function that processes the values from that field into a categorical
variable.

.. code:: python

    "additional_weights": {
        "month": {"field_lookup": "date", "mapper": lambda x: "{}_{}".format(x.year, x.month),
        "new_politician": {"field_lookup": "politician_id", "mapper": lambda x: str(Politician.objects.get(pk=x).terms.count() == 1)
    }


Built-in methods
-----------------

``all_documents``
******************

Simply creates a "sample" containing all of the documents in the sampling frame.

``random``
************

Simple random sampling.