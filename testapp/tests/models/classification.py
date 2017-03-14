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
        model = get_test_model("test")
        test_scores = model.get_test_prediction_results()
        fold_scores = model.get_cv_prediction_results()

        self.assertEqual(len(model.dataset), 200)
        self.assertEqual(len(model.train_dataset), 160)
        self.assertEqual(len(model.test_dataset), 40)
        self.assertEqual(len(model.cv_folds), 5)

        for a, b in model.cv_folds:
            self.assertGreaterEqual(len(a), 127)
            self.assertLessEqual(len(a), 129)
            self.assertGreaterEqual(len(b), 31)
            self.assertLessEqual(len(b), 33)
        self.assertEqual(list(set(test_scores["n"].values))[0], 40)
        self.assertEqual(list(set(fold_scores["n"].values))[0], 32)

        pos_label = (
            Project.objects.get(name="test_project")
            .questions.get(name="test_checkbox")
            .labels.get(value="1")
            .pk
        )
        neg_label = (
            Project.objects.get(name="test_project")
            .questions.get(name="test_checkbox")
            .labels.get(value="0")
            .pk
        )
        for label_id, metric, expected_val in [
            (pos_label, "precision", 1.0),
            (pos_label, "recall", 0.6),
            (pos_label, "n", 40),
            (pos_label, "matthews_corrcoef", 0.75),
            (pos_label, "cohens_kappa", 0.73),
            (pos_label, "accuracy", 0.95),
            (neg_label, "precision", 0.95),
            (neg_label, "recall", 1.0),
            (neg_label, "n", 40),
            (neg_label, "matthews_corrcoef", 0.75),
            (neg_label, "cohens_kappa", 0.72),
            (neg_label, "accuracy", 0.95),
        ]:
            val = test_scores.loc[
                test_scores["outcome_column"] == "label_id__{}".format(label_id)
            ][metric].values[0]
            print("{}: {}".format(metric, val))
            self.assertGreater(val, expected_val - 0.01)
            self.assertLess(val, expected_val + 0.01)

        incorrect = model.get_incorrect_predictions()
        self.assertGreaterEqual(len(incorrect), 4)
        self.assertLessEqual(len(incorrect), 8)

        model.set_probability_threshold(0.5)
        model.apply_model_to_documents(Document.objects.all())
        self.assertEqual(model.classifications.count(), 250)
        neg_count = model.classifications.filter(label_id=neg_label).count()
        pos_count = model.classifications.filter(label_id=pos_label).count()

        self.assertTrue(218 <= neg_count <= 224)
        self.assertTrue(26 <= pos_count <= 32)

        model.set_probability_threshold(0.25)
        model.update_classifications_with_probability_threshold()
        self.assertLessEqual(
            model.classifications.filter(label_id=neg_label).count(), neg_count
        )
        self.assertGreaterEqual(
            model.classifications.filter(label_id=pos_label).count(), pos_count
        )

        model.set_probability_threshold(0.99)
        model.update_classifications_with_probability_threshold()
        self.assertGreaterEqual(
            model.classifications.filter(label_id=neg_label).count(), neg_count
        )
        self.assertLessEqual(
            model.classifications.filter(label_id=pos_label).count(), pos_count
        )

        preds = model.produce_prediction_dataset(
            model.dataset[:10], ignore_probability_threshold=True
        )
        self.assertEqual(len(preds), 10)
        self.assertIn("label_id", preds.columns)
        self.assertIn("probability", preds.columns)

        model.apply_model_to_frame(
            save=True, document_filters=None, refresh=True, num_cores=1
        )

    def test_document_classification_model_with_balancing_variables(self):

        set_up_test_sample("test_sample", 200)
        model = get_test_model("test_with_balancing_variables")
        test_scores = model.get_test_prediction_results()
        fold_scores = model.get_cv_prediction_results()

        self.assertGreater(model.dataset["balancing_weight"].nunique(), 1)
        self.assertGreater(model.dataset["training_weight"].nunique(), 1)

        self.assertEqual(len(model.dataset), 200)
        self.assertEqual(len(model.train_dataset), 160)
        self.assertEqual(len(model.test_dataset), 40)
        self.assertEqual(len(model.cv_folds), 5)

        for a, b in model.cv_folds:
            self.assertGreaterEqual(len(a), 127)
            self.assertLessEqual(len(a), 129)
            self.assertGreaterEqual(len(b), 31)
            self.assertLessEqual(len(b), 33)
        self.assertEqual(list(set(test_scores["n"].values))[0], 40)
        self.assertEqual(list(set(fold_scores["n"].values))[0], 32)

        pos_label = (
            Project.objects.get(name="test_project")
            .questions.get(name="test_checkbox")
            .labels.get(value="1")
            .pk
        )
        neg_label = (
            Project.objects.get(name="test_project")
            .questions.get(name="test_checkbox")
            .labels.get(value="0")
            .pk
        )
        for label_id, metric, expected_val in [
            (pos_label, "precision", 1.0),
            (pos_label, "recall", 0.6),
            (pos_label, "n", 40),
            (pos_label, "matthews_corrcoef", 0.75),
            (pos_label, "cohens_kappa", 0.73),
            (pos_label, "accuracy", 0.95),
            (neg_label, "precision", 0.95),
            (neg_label, "recall", 1.0),
            (neg_label, "n", 40),
            (neg_label, "matthews_corrcoef", 0.75),
            (neg_label, "cohens_kappa", 0.72),
            (neg_label, "accuracy", 0.95),
        ]:
            val = test_scores.loc[
                test_scores["outcome_column"] == "label_id__{}".format(label_id)
            ][metric].values[0]
            print("{}: {}".format(metric, val))
            self.assertGreater(val, expected_val - 0.01)
            self.assertLess(val, expected_val + 0.01)

        incorrect = model.get_incorrect_predictions()
        self.assertGreaterEqual(len(incorrect), 4)
        self.assertLessEqual(len(incorrect), 8)
        model.apply_model_to_documents(Document.objects.all())
        self.assertEqual(model.classifications.count(), 250)
        neg_count = model.classifications.filter(label_id=neg_label).count()
        pos_count = model.classifications.filter(label_id=pos_label).count()

        self.assertTrue(228 <= neg_count <= 234)
        self.assertTrue(16 <= pos_count <= 22)

        model.set_probability_threshold(0.25)
        model.update_classifications_with_probability_threshold()
        self.assertLessEqual(
            model.classifications.filter(label_id=neg_label).count(), neg_count
        )
        self.assertGreaterEqual(
            model.classifications.filter(label_id=pos_label).count(), pos_count
        )

        model.set_probability_threshold(0.99)
        model.update_classifications_with_probability_threshold()
        self.assertGreaterEqual(
            model.classifications.filter(label_id=neg_label).count(), neg_count
        )
        self.assertLessEqual(
            model.classifications.filter(label_id=pos_label).count(), pos_count
        )

        preds = model.produce_prediction_dataset(
            model.dataset[:10], ignore_probability_threshold=True
        )
        self.assertEqual(len(preds), 10)
        self.assertIn("label_id", preds.columns)
        self.assertIn("probability", preds.columns)

        model.apply_model_to_frame(
            save=True, document_filters=None, refresh=True, num_cores=1
        )

    def test_document_classification_model_with_holdout(self):

        set_up_test_sample("test_sample", 160)
        set_up_test_sample("test_sample_holdout", 40)
        model = get_test_model("test_with_holdout")
        test_scores = model.get_test_prediction_results()

        self.assertEqual(len(model.dataset), 200)
        self.assertEqual(len(model.train_dataset), 160)
        self.assertEqual(len(model.test_dataset), 40)
        self.assertEqual(list(set(test_scores["n"].values))[0], 40)

        pos_label = (
            Project.objects.get(name="test_project")
            .questions.get(name="test_checkbox")
            .labels.get(value="1")
            .pk
        )
        neg_label = (
            Project.objects.get(name="test_project")
            .questions.get(name="test_checkbox")
            .labels.get(value="0")
            .pk
        )
        for label_id, metric, expected_val in [
            (pos_label, "precision", 1.0),
            (pos_label, "recall", 0.75),
            (pos_label, "n", 40),
            (pos_label, "matthews_corrcoef", 0.85),
            (pos_label, "cohens_kappa", 0.84),
            (pos_label, "accuracy", 0.975),
            (neg_label, "precision", 0.97),
            (neg_label, "recall", 1.0),
            (neg_label, "n", 40),
            (neg_label, "matthews_corrcoef", 0.85),
            (neg_label, "cohens_kappa", 0.84),
            (neg_label, "accuracy", 0.975),
        ]:
            val = test_scores.loc[
                test_scores["outcome_column"] == "label_id__{}".format(label_id)
            ][metric].values[0]
            print("{}: {}".format(metric, val))
            self.assertGreater(val, expected_val - 0.01)
            self.assertLess(val, expected_val + 0.01)

        incorrect = model.get_incorrect_predictions()
        self.assertGreaterEqual(len(incorrect), 2)
        self.assertLessEqual(len(incorrect), 6)
        model.apply_model_to_documents(Document.objects.all())
        self.assertEqual(model.classifications.count(), 250)
        neg_count = model.classifications.filter(label_id=neg_label).count()
        pos_count = model.classifications.filter(label_id=pos_label).count()

        self.assertTrue(222 <= neg_count <= 228)
        self.assertTrue(22 <= pos_count <= 28)

        model.set_probability_threshold(0.25)
        model.update_classifications_with_probability_threshold()
        self.assertLessEqual(
            model.classifications.filter(label_id=neg_label).count(), neg_count
        )
        self.assertGreaterEqual(
            model.classifications.filter(label_id=pos_label).count(), pos_count
        )

        model.set_probability_threshold(1.0)
        model.update_classifications_with_probability_threshold()
        self.assertGreaterEqual(
            model.classifications.filter(label_id=neg_label).count(), neg_count
        )
        self.assertLessEqual(
            model.classifications.filter(label_id=pos_label).count(), pos_count
        )

        preds = model.produce_prediction_dataset(
            model.dataset[:10], ignore_probability_threshold=True
        )
        self.assertEqual(len(preds), 10)
        self.assertIn("label_id", preds.columns)
        self.assertIn("probability", preds.columns)

        model.apply_model_to_frame(
            save=True, document_filters=None, refresh=True, num_cores=1
        )

    def test_document_classification_model_with_keyword_oversample(self):

        set_up_test_sample("test_sample_keyword_oversample", 200)
        model = get_test_model("test_with_keyword_oversample")
        test_scores = model.get_test_prediction_results()
        fold_scores = model.get_cv_prediction_results()

        self.assertGreater(model.dataset["training_weight"].nunique(), 1)

        self.assertEqual(len(model.dataset), 200)
        self.assertEqual(len(model.train_dataset), 160)
        self.assertEqual(len(model.test_dataset), 40)
        self.assertEqual(len(model.cv_folds), 5)

        for a, b in model.cv_folds:
            self.assertGreaterEqual(len(a), 127)
            self.assertLessEqual(len(a), 129)
            self.assertGreaterEqual(len(b), 31)
            self.assertLessEqual(len(b), 33)
        self.assertEqual(list(set(test_scores["n"].values))[0], 40)
        self.assertEqual(list(set(fold_scores["n"].values))[0], 32)

        pos_label = (
            Project.objects.get(name="test_project")
            .questions.get(name="test_checkbox")
            .labels.get(value="1")
            .pk
        )
        neg_label = (
            Project.objects.get(name="test_project")
            .questions.get(name="test_checkbox")
            .labels.get(value="0")
            .pk
        )
        for label_id, metric, expected_val in [
            (pos_label, "precision", 1.0),
            (pos_label, "recall", 1.0),
            (pos_label, "n", 40),
            (pos_label, "matthews_corrcoef", 1.0),
            (pos_label, "cohens_kappa", 1.0),
            (pos_label, "accuracy", 1.0),
            (neg_label, "precision", 1.0),
            (neg_label, "recall", 1.0),
            (neg_label, "n", 40),
            (neg_label, "matthews_corrcoef", 1.0),
            (neg_label, "cohens_kappa", 1.0),
            (neg_label, "accuracy", 1.0),
        ]:
            val = test_scores.loc[
                test_scores["outcome_column"] == "label_id__{}".format(label_id)
            ][metric].values[0]
            print("{}: {}".format(metric, val))
            self.assertGreater(val, expected_val - 0.01)
            self.assertLess(val, expected_val + 0.01)

        incorrect = model.get_incorrect_predictions()
        self.assertGreaterEqual(len(incorrect), 2)
        self.assertLessEqual(len(incorrect), 8)
        model.apply_model_to_documents(Document.objects.all())
        self.assertEqual(model.classifications.count(), 250)
        neg_count = model.classifications.filter(label_id=neg_label).count()
        pos_count = model.classifications.filter(label_id=pos_label).count()

        self.assertTrue(224 <= neg_count <= 228)
        self.assertTrue(22 <= pos_count <= 26)

        model.set_probability_threshold(0.25)
        model.update_classifications_with_probability_threshold()
        self.assertLessEqual(
            model.classifications.filter(label_id=neg_label).count(), neg_count
        )
        self.assertGreaterEqual(
            model.classifications.filter(label_id=pos_label).count(), pos_count
        )

        model.set_probability_threshold(0.99)
        model.update_classifications_with_probability_threshold()
        self.assertGreaterEqual(
            model.classifications.filter(label_id=neg_label).count(), neg_count
        )
        self.assertLessEqual(
            model.classifications.filter(label_id=pos_label).count(), pos_count
        )

        preds = model.produce_prediction_dataset(
            model.dataset[:10], ignore_probability_threshold=True
        )
        self.assertEqual(len(preds), 10)
        self.assertIn("label_id", preds.columns)
        self.assertIn("probability", preds.columns)

        model.apply_model_to_frame(
            save=True, document_filters=None, refresh=True, num_cores=1
        )

    def tearDown(self):

        from django.conf import settings
        import shutil

        shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
