Topic models
=============

Topic models are defined in config files that should be placed in one of your ``settings.DJANGO_LEARNING_TOPIC_MODELS``
folders. The files should have a ``get_parameters`` function that returns a dictionary of parameters that specify
how the model should be trained. All topic models in Django Learning use the CorEx algorithm, which allows you to
iteratively train your model using anchor terms.

.. code:: python

    def get_parameters():

        return {
            "frame": "my_frame",
            "num_topics": 10,
            "sample_size": 50,
            "anchor_strength": 4,
            "vectorizer": {
                "sublinear_tf": False,
                "max_df": 0.9,
                "min_df": 5,
                "max_features": 8000,
                "ngram_range": (1, 3),
                "use_idf": False,
                "norm": None,
                "binary": True,
                "preprocessors": [
                    (
                        "clean_text",
                        {
                            "process_method": ["lemmatize"],
                            "regex_filters": [],
                            "stopword_sets": ["english"],
                            "stopword_whitelists": [],
                        },
                    )
                ],
            },
        }

Parameters
-----------

Topic models should specify the following parameters:

    - "frame": name of a sampling frame of documents
    - "num_topics": the number of topics to extract
    - "sample_size": number of documents to train the model on (random sample)
    - "anchor_strength": strength of the anchor terms (see CorEx documentation)
    - "vectorizer": parameters to pass to the ``tfidf`` feature extractor, which will be used to vectorize the documents