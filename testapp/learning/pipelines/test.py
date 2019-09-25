import numpy as np

from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.impute import SimpleImputer

from django_pewtils import get_model
from django_learning.utils.feature_extractors import feature_extractors


def get_pipeline():

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
            "parameters": {
                "project_name": "test_project",
                "sandbox": True,
                "sample_names": ["test_sample"],
                "question_names": ["test_checkbox"],
                "document_filters": [],
                "coder_filters": [("exclude_mturk", [], {})],
                "base_class_id": base_class_id,
                "threshold": 0.4,
                "convert_to_discrete": True,
                "balancing_variables": [],
                "ignore_stratification_weights": False,
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
            "test_percent": 0.25,
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
                    "min_df": [10],
                    "max_features": [None],
                    "ngram_range": [[1, 4]],
                    "use_idf": [True],
                    "norm": ["l2"],
                    "preprocessors": [
                        [
                            (
                                "clean_text",
                                {
                                    "lemmatize": True,
                                    "regex_filters": [],
                                    "stopword_sets": ["english", "test"],
                                    "stopword_whitelists": ["test"],
                                },
                            )
                        ]
                    ],
                },
                "tfidf_bool": {
                    "sublinear_tf": [False],
                    "max_df": [0.9],
                    "min_df": [10],
                    "max_features": [None],
                    "ngram_range": [[1, 4]],
                    "use_idf": [False],
                    "norm": [None],
                    "preprocessors": [
                        [
                            (
                                "clean_text",
                                {
                                    "lemmatize": True,
                                    "regex_filters": [],
                                    "stopword_sets": ["english", "test"],
                                    "stopword_whitelists": ["test"],
                                },
                            )
                        ]
                    ],
                },
            },
        },
    }
