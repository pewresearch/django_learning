from __future__ import absolute_import

import numpy as np

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer

from django_pewtils import get_model

from testapp.utils import get_base_dataset_parameters


def get_pipeline():

    from django_learning.utils.feature_extractors import feature_extractors

    base_class_id = (
        get_model("Question", app_name="django_learning")
        .objects.filter(project__name="test_project")
        .get(name="test_checkbox")
        .labels.get(value="0")
        .pk
    )

    return {
        "dataset_extractor": {
            "name": "document_dataset",
            "parameters": get_base_dataset_parameters(
                "document_dataset", params={"balancing_variables": ["test"]}
            ),
            "outcome_column": "label_id",
        },
        "model": {
            "name": "classification_xgboost",
            "cv": 5,
            "params": {},
            "fit_params": {"eval_metric": "error"},
            "use_sample_weights": True,
            "use_class_weights": False,
            "test_percent": 0.2,
            "scoring_function": "maxmin",
        },
        "pipeline": {
            "steps": [
                (
                    "features",
                    FeatureUnion(
                        [
                            ("tfidf_counts", feature_extractors["tfidf"]()),
                            ("tfidf_bool", feature_extractors["tfidf"]()),
                        ]
                    ),
                ),
                ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
            ],
            "params": {
                "tfidf_counts": {
                    "sublinear_tf": [False],
                    "max_df": [0.9],
                    "min_df": [5, 10],
                    "max_features": [None],
                    "ngram_range": [[1, 4]],
                    "use_idf": [True],
                    "norm": ["l2"],
                    "preprocessors": [
                        [
                            (
                                "clean_text",
                                {
                                    "process_method": "lemmatize",
                                    "regex_filters": [],
                                    "stopword_sets": ["english", "test"],
                                    "stopword_whitelists": ["test"],
                                    "refresh_stopwords": False,
                                },
                            )
                        ]
                    ],
                },
                "tfidf_bool": {
                    "sublinear_tf": [False],
                    "max_df": [0.9],
                    "min_df": [5],
                    "max_features": [None],
                    "ngram_range": [[1, 4]],
                    "use_idf": [False],
                    "norm": [None],
                    "preprocessors": [
                        [
                            (
                                "clean_text",
                                {
                                    "process_method": "lemmatize",
                                    "regex_filters": [],
                                    "stopword_sets": ["english", "test"],
                                    "stopword_whitelists": ["test"],
                                    "refresh_stopwords": False,
                                },
                            )
                        ]
                    ],
                },
            },
        },
    }
