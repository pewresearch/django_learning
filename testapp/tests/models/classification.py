from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project, set_up_test_sample, get_test_model


class ClassificationTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project(limit=250)

    def test_document_classification_model(self):

        set_up_test_sample("test_sample", 200)
        model = get_test_model("test_model")
        test_scores = model.get_test_prediction_results()
        fold_scores = model.get_cv_prediction_results()

        self.assertEqual(len(model.dataset), 200)
        self.assertEqual(len(model.train_dataset), 150)
        self.assertEqual(len(model.test_dataset), 50)
        self.assertEqual(len(model.cv_folds), 5)
        for a, b in model.cv_folds:
            self.assertGreaterEqual(len(a), 119)
            self.assertLessEqual(len(a), 121)
            self.assertGreaterEqual(len(b), 29)
            self.assertLessEqual(len(b), 31)
        self.assertEqual(list(set(test_scores["n"].values))[0], 50)
        self.assertEqual(list(set(fold_scores["n"].values))[0], 30)

        for label_id, metric, expected_val in [
            (11, "precision", 1.0),
            (11, "recall", 0.6),
            (11, "n", 50),
            (11, "matthews_corrcoef", 0.76),
            (11, "cohens_kappa", 0.73),
            (11, "accuracy", 0.96),
            (12, "precision", 0.96),
            (12, "recall", 1.0),
            (12, "n", 50),
            (12, "matthews_corrcoef", 0.76),
            (12, "cohens_kappa", 0.73),
            (12, "accuracy", 0.96),
        ]:
            val = test_scores.loc[
                test_scores["outcome_column"] == "label_id__{}".format(label_id)
            ][metric].values[0]
            self.assertGreater(val, expected_val - 0.01)
            self.assertLess(val, expected_val + 0.01)

        import pdb

        pdb.set_trace()
        incorrect = model.get_incorrect_predictions()
        self.assertGreaterEqual(len(incorrect), 7)
        self.assertLessEqual(len(incorrect), 9)
        model.apply_model_to_documents(Document.objects.all())
        self.assertEqual(model.classifications.count(), 250)
        self.assertGreaterEqual(model.classifications.filter(label_id=12).count(), 229)
        self.assertLessEqual(model.classifications.filter(label_id=12).count(), 233)
        self.assertGreaterEqual(model.classifications.filter(label_id=11).count(), 17)
        self.assertLessEqual(model.classifications.filter(label_id=11).count(), 21)
        model.set_probability_threshold(0.4)
        model.update_classifications_with_probability_threshold()
        self.assertGreaterEqual(model.classifications.filter(label_id=12).count(), 221)
        self.assertLessEqual(model.classifications.filter(label_id=12).count(), 225)
        self.assertGreaterEqual(model.classifications.filter(label_id=11).count(), 25)
        self.assertLessEqual(model.classifications.filter(label_id=11).count(), 29)

        import pdb

        pdb.set_trace()
        model.set_probability_threshold(0.85)
        preds = model.produce_prediction_dataset(
            model.dataset[:10], ignore_probability_threshold=True
        )
        self.assertEqual(len(preds), 10)
        self.assertIn("label_id", preds.columns)
        self.assertIn("probability", preds.columns)
        preds = model.produce_prediction_dataset(
            model.dataset[:10], ignore_probability_threshold=False
        )
        self.assertEqual(len(preds), 10)
        self.assertIn("label_id", preds.columns)
        self.assertIn("probability", preds.columns)
        # model.apply_model_to_frame(save=True, document_filters=None, refresh=False, num_cores=2, chunk_size=1000)

        #
        #

        # predicted_df = dataset_extractors["model_prediction_dataset"](
        #     dataset=df_to_predict,
        #     learning_model=self,
        #     cache_key=cache_key,
        #     disable_probability_threshold_warning=disable_probability_threshold_warning,
        # ).extract(refresh=refresh, only_load_existing=only_load_existing)

        # TODO: test the `model_prediction_dataset` dataset extractor

    # def test_document_classification_model_with_holdout(self):
    #     set_up_test_sample("test_sample_holdout", 100)
    #     frame = SamplingFrame.objects.get(name="all_documents")
    #
    #     ### Classifier with holdout
    #     model = DocumentClassificationModel.objects.create(
    #         name="test_model_with_holdout",
    #         pipeline_name="test_with_holdout",
    #         sampling_frame=frame,
    #     )
    #     model.extract_dataset(refresh=True)
    #     model.load_model(refresh=True, num_cores=1)
    #     model.describe_model()
    #     model.get_test_prediction_results(refresh=True)
    #     model.find_probability_threshold(save=True)
    #     test_scores = model.get_test_prediction_results()
    #
    #     self.assertEqual(len(model.dataset), 1100)
    #     self.assertEqual(len(model.train_dataset), 1000)
    #     self.assertEqual(len(model.test_dataset), 100)
    #     self.assertEqual(list(set(test_scores["n"].values))[0], 100)
    #
    #     for label_id, metric, expected_val in [
    #         (11, "precision", 1.0),
    #         (11, "recall", 0.82),
    #         (11, "n", 100),
    #         (11, "matthews_corrcoef", 0.89),
    #         (11, "cohens_kappa", 0.89),
    #         (11, "accuracy", 0.98),
    #         (12, "precision", 0.98),
    #         (12, "recall", 1.0),
    #         (12, "n", 100),
    #         (12, "matthews_corrcoef", 0.89),
    #         (12, "cohens_kappa", 0.89),
    #         (12, "accuracy", 0.98),
    #     ]:
    #         val = test_scores.loc[
    #             test_scores["outcome_column"] == "label_id__{}".format(label_id)
    #         ][metric].values[0]
    #         # print("{}: {} ({} expected".format(metric, val, expected_val))
    #         self.assertAlmostEqual(val, expected_val, 2)

    def tearDown(self):

        from django.conf import settings
        import shutil

        shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
