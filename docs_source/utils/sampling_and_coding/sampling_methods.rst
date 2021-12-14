Sampling methods
=================

Sampling methods specify a way to pull a sample of documents from a frame. They are defined in sampling method config
files which require a ``get_method`` function that returns a dictionary with the following parameters:

.. code:: python

    def get_method():

        return {
            "sampling_strategy": "random",
            "stratify_by": None,
            "sampling_searches": [{"regex_filter": "test", "proportion": 0.5}],
        }

Parameters

sampling_strategy

stratify_by

sampling_searches




Built-in methods

``all_documents``

Simply creates a "sample" containing all of the documents in the sampling frame.

``random``

Simple random sampling.