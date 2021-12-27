from __future__ import print_function
import datetime
import pandas as pd
import numpy as np

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands
from pewanalytics.stats.irr import compute_overall_scores
from pewtils import is_not_null

from testapp.models import MovieReview
from testapp.utils import set_up_test_project, set_up_test_sample


class ScoringTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()
        set_up_test_sample("test_sample", 200)

    def test_compute_scores_from_dataset(self):

        from django_learning.utils.dataset_extractors import dataset_extractors
        from django_learning.utils.scoring import compute_scores_from_dataset

        ### MULTIPLE QUESTIONS

        extractor = dataset_extractors["document_coder_label_dataset"](
            **{
                "project_name": "test_project",
                "sample_names": ["test_sample"],
                "question_names": ["test_checkbox", "test_radio"],
            }
        )
        dataset = extractor.extract(refresh=True)

        scores = compute_scores_from_dataset(
            dataset,
            "document_id",
            "label_id",
            "coder_id",
            weight_column="sampling_weight",
            discrete_classes=True,
            pos_label=None,
        )
        for metric, values in [
            (
                "alpha_unweighted",
                [
                    0.9114251886932071,
                    0.9364548494983278,
                    0.79,
                    1.0,
                    0.9445833333333333,
                    -0.002512562814070529,
                ],
            ),
            ("precision", [0.97, 0.9875776397515528, 0.8, 1.0, 0.95, 0.0]),
            ("recall", [0.97, 0.9875776397515528, 0.8, 1.0, 0.95, 0.0]),
            (
                "matthews_corrcoef",
                [
                    0.9112031966849193,
                    0.9362955884695016,
                    0.7894736842105263,
                    1.0,
                    0.9444444444444444,
                    -0.005025125628140704,
                ],
            ),
            (
                "roc_auc",
                [
                    0.9681477942347507,
                    0.8947368421052632,
                    1.0,
                    0.9722222222222222,
                    0.49748743718592964,
                ],
            ),
            ("pct_agree_unweighted", [0.97, 0.98, 0.98, 1.0, 0.99, 0.99]),
            (
                "cohens_kappa",
                [
                    0.9362955884695014,
                    0.7894736842105263,
                    1.0,
                    0.9444444444444444,
                    -0.005025125628140614,
                ],
            ),
        ]:
            self.assertEqual(
                [s for s in list(scores[metric].values) if is_not_null(s)], values
            )

        overall = compute_overall_scores(dataset, "label_id", "document_id", "coder_id")
        self.assertEqual(overall["alpha"], 0.9114251886932071)
        self.assertEqual(overall["fleiss_kappa"], 0.9112031966849192)

        ### SINGLE QUESTION

        extractor = dataset_extractors["document_coder_label_dataset"](
            **{
                "project_name": "test_project",
                "sample_names": ["test_sample"],
                "question_names": ["test_checkbox"],
            }
        )
        dataset = extractor.extract(refresh=True)
        scores = compute_scores_from_dataset(
            dataset, "document_id", "label_id", "coder_id"
        )
        scores = scores.set_index("outcome_column").to_dict("index")["label_id__10"]
        for metric, score in [
            ("cohens_kappa", 0.7113997113997114),
            ("precision", 0.7272727272727273),
            ("recall", 0.7272727272727273),
            ("alpha_unweighted", 0.7121212121212122),
        ]:
            self.assertEqual(scores[metric], score)

        ### SINGLE QUESTION BINARY

        extractor = dataset_extractors["document_coder_dataset"](
            **{
                "project_name": "test_project",
                "sample_names": ["test_sample"],
                "question_names": ["test_checkbox"],
            }
        )
        dataset = extractor.extract(refresh=True)
        scores = compute_scores_from_dataset(
            dataset,
            "document_id",
            "label_10",
            "coder_id",
            weight_column="sampling_weight",
            pos_label=1,
        )
        scores = scores.to_dict("records")[0]
        for metric, score in [
            ("cohens_kappa", 0.7113997113997114),
            ("precision", 0.7272727272727273),
            ("recall", 0.7272727272727273),
            ("alpha_unweighted", 0.7121212121212122),
        ]:
            self.assertEqual(scores[metric], score)

    def test_compute_scores_from_datasets_as_coders(self):

        from django_learning.utils.dataset_extractors import dataset_extractors
        from django_learning.utils.scoring import compute_scores_from_datasets_as_coders

        pos_class_id = (
            Question.objects.get(name="test_checkbox").labels.get(value="1").pk
        )
        base_class_id = (
            Question.objects.get(name="test_checkbox").labels.get(value="0").pk
        )
        low_threshold = dataset_extractors["document_dataset"](
            **{
                "project_name": "test_project",
                "sample_names": ["test_sample"],
                "question_names": ["test_checkbox"],
                "coder_aggregation_function": "mean",
                "convert_to_discrete": True,
                "threshold": 0.0,
                "base_class_id": base_class_id,
            }
        ).extract(refresh=True)
        high_threshold = dataset_extractors["document_dataset"](
            **{
                "project_name": "test_project",
                "sample_names": ["test_sample"],
                "question_names": ["test_checkbox"],
                "coder_aggregation_function": "mean",
                "convert_to_discrete": True,
                "threshold": 1.0,
                "base_class_id": base_class_id,
            }
        ).extract(refresh=True)

        scores = compute_scores_from_datasets_as_coders(
            low_threshold,
            high_threshold,
            "document_id",
            "label_id",
            weight_column="sampling_weight",
            min_overlap=10,
            discrete_classes=True,
        )
        scores = scores.set_index("outcome_column").to_dict("index")[
            "label_id__{}".format(int(pos_class_id))
        ]
        for metric, score in [
            ("cohens_kappa", 0.7126436781609196),
            ("precision", 1.0),
            ("recall", 0.5714285714285714),
            ("alpha_unweighted", 0.7121212121212122),
        ]:
            self.assertEqual(scores[metric], score)

    # TODO: test scoring utils
    #   get_probability_threshold_score_df
    #   get_probability_threshold_from_score_df
    #   find_probability_threshold
    #   apply_probability_threshold

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
