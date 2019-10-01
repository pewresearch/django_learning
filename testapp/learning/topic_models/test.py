from __future__ import absolute_import


def get_parameters():

    return {
        "frame": "test",
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
                        "lemmatize": True,
                        "regex_filters": [],
                        "stopword_sets": ["english", "test"],
                        "stopword_whitelists": ["test"],
                    },
                )
            ],
        },
    }
