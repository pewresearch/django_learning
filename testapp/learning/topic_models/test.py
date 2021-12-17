from __future__ import absolute_import


def get_parameters():

    return {
        "frame": "all_documents",
        "num_topics": 5,
        "sample_size": 1000,
        "anchor_strength": 4,
        "vectorizer": {
            "sublinear_tf": False,
            "max_df": 0.9,
            "min_df": 10,
            "max_features": 8000,
            "ngram_range": (1, 2),
            "use_idf": False,
            "norm": None,
            "binary": True,
            "preprocessors": [
                (
                    "clean_text",
                    {
                        "process_method": ["lemmatize"],
                        "regex_filters": [],
                        "stopword_sets": ["english", "test"],
                        "stopword_whitelists": ["test"],
                    },
                )
            ],
        },
    }
