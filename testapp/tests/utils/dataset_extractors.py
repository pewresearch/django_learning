from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import *


class DatasetExtractorsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_sample("test_sample", 100)

    def test_loading(self):

        from django_learning.utils.dataset_extractors import dataset_extractors

        for val in [
            "document_coder_dataset",
            "document_coder_label_dataset",
            "document_dataset",
            "model_prediction_dataset",
            "raw_document_dataset",
        ]:
            self.assertIn(val, dataset_extractors.keys())
            self.assertIsNotNone(dataset_extractors[val])

    def test_document_coder_label_dataset_extractor(self):

        from django_learning.utils.dataset_extractors import dataset_extractors

        params = get_base_dataset_parameters("document_coder_label_dataset")
        extractor = dataset_extractors["document_coder_label_dataset"](**params)
        scores = extractor.compute_scores(
            refresh=True, min_overlap=5, discrete_classes=True
        )
        self.assertAlmostEqual(scores["cohens_kappa"].mean(), 0.7777, 3)
        scores = extractor.compute_overall_scores()
        self.assertAlmostEqual(scores["alpha"], 0.7788, 3)
        self.assertAlmostEqual(scores["fleiss_kappa"], 0.7777, 3)

        df = extractor.extract()
        self.assertEqual(len(df), 200)
        self.assertEqual(df["document_id"].nunique(), 100)
        self.assertEqual(df["coder_id"].nunique(), 2)

    def test_document_coder_dataset_extractor(self):

        label_pos = Label.objects.filter(question__name="test_checkbox").get(value="1")
        label_neg = Label.objects.filter(question__name="test_checkbox").get(value="0")
        df = extract_dataset(
            "document_coder_dataset", params={"standardize_coders": False}
        )
        self.assertEqual(df["label_{}".format(label_pos.pk)].nunique(), 2)
        self.assertEqual(df["label_{}".format(label_pos.pk)].max(), 1)
        self.assertEqual(df["label_{}".format(label_pos.pk)].min(), 0)
        self.assertEqual(df["label_{}".format(label_neg.pk)].nunique(), 2)
        self.assertEqual(df["label_{}".format(label_neg.pk)].max(), 1)
        self.assertEqual(df["label_{}".format(label_neg.pk)].min(), 0)

        df = extract_dataset(
            "document_coder_dataset", params={"standardize_coders": True}
        )
        self.assertAlmostEqual(
            df[df["coder_name"] == "coder1"]["label_{}".format(label_pos.pk)].mean(),
            0.0,
            2,
        )
        self.assertAlmostEqual(
            df[df["coder_name"] == "coder1"]["label_{}".format(label_neg.pk)].mean(),
            0.0,
            2,
        )
        self.assertAlmostEqual(
            df[df["coder_name"] == "coder2"]["label_{}".format(label_pos.pk)].mean(),
            0.0,
            2,
        )
        self.assertAlmostEqual(
            df[df["coder_name"] == "coder2"]["label_{}".format(label_neg.pk)].mean(),
            0.0,
            2,
        )

    def test_document_dataset_extractor(self):

        label_pos = Label.objects.filter(question__name="test_checkbox").get(value="1")
        label_neg = Label.objects.filter(question__name="test_checkbox").get(value="0")
        for update_params, values in [
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": False,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                (88, 4, 8, 8, 4, 88),
            ),
            (
                {
                    "coder_aggregation_function": "max",
                    "convert_to_discrete": False,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                (88, 0, 12, 8, 0, 92),
            ),
            (
                {
                    "coder_aggregation_function": "min",
                    "convert_to_discrete": False,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                (92, 0, 8, 12, 0, 88),
            ),
            (
                {
                    "coder_aggregation_function": "median",
                    "convert_to_discrete": False,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                (88, 4, 8, 8, 4, 88),
            ),
        ]:
            df = extract_dataset("document_dataset", params=update_params)
            # if len(extractor.outcome_columns) == 2:
            # if "label_{}".format(label_pos.pk) not in df.columns:
            # print(
            #     "{}, {}, {}, {}, {}, {}".format(
            #         len(df[df["label_{}".format(label_pos.pk)] == 0.0]),
            #         len(df[df["label_{}".format(label_pos.pk)] == 0.5]),
            #         len(df[df["label_{}".format(label_pos.pk)] == 1.0]),
            #         len(df[df["label_{}".format(label_neg.pk)] == 0.0]),
            #         len(df[df["label_{}".format(label_neg.pk)] == 0.5]),
            #         len(df[df["label_{}".format(label_neg.pk)] == 1.0]),
            #     )
            # )
            self.assertEqual(
                len(df[df["label_{}".format(label_pos.pk)] == 0.0]), values[0]
            )
            self.assertEqual(
                len(df[df["label_{}".format(label_pos.pk)] == 0.5]), values[1]
            )
            self.assertEqual(
                len(df[df["label_{}".format(label_pos.pk)] == 1.0]), values[2]
            )
            self.assertEqual(
                len(df[df["label_{}".format(label_neg.pk)] == 0.0]), values[3]
            )
            self.assertEqual(
                len(df[df["label_{}".format(label_neg.pk)] == 0.5]), values[4]
            )
            self.assertEqual(
                len(df[df["label_{}".format(label_neg.pk)] == 1.0]), values[5]
            )

        for update_params, values in [
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 1.0,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                {str(label_neg.pk): 88, str(label_pos.pk): 8, "None": 4},
            ),
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 0.0,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                {str(label_neg.pk): 88, str(label_pos.pk): 12},
            ),
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 0.5,
                    "base_class_id": label_pos.pk,
                    "question_names": ["test_checkbox"],
                },
                {str(label_neg.pk): 92, str(label_pos.pk): 8},
            ),
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 0.4,
                    "base_class_id": None,
                    "question_names": ["test_checkbox", "test_radio"],
                },
                {"0101": 89, "1001": 3, "0110": 2, "1010": 6},
                # TODO: investigate how the above function is working
                # for multiple codes, does it concatenate first and then collapse with the threshold?
                # or does it collapse each item separately with the threshold and then concatenate?
                # and what happens if there's a tie and you don't have a base class specified?
            ),
        ]:
            df = extract_dataset("document_dataset", params=update_params)
            vals = df["label_id"].value_counts().to_dict()
            self.assertEqual(vals, values)

    def test_raw_document_dataset(self):

        extractor = dataset_extractors["raw_document_dataset"](
            document_ids=[], sampling_frame_name="all_documents", document_filters=[]
        )
        df = extractor.extract()
        self.assertEqual(len(df), 150)
        self.assertIn("text", df.columns)
        self.assertIn("date", df.columns)

    def test_model_prediction_dataset(self):
        # See: tests/models/classification.py
        pass

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
