from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import *


class DatasetDocumentFiltersTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        pass

    def test_loading(self):
        from django_learning.utils.dataset_document_filters import (
            dataset_document_filters,
        )

        for val in [
            "django_lookup_filter",
            "filter_by_date",
            "filter_by_document_ids",
            "filter_by_other_model_dataset",
            "filter_by_other_model_prediction",
        ]:
            self.assertIn(val, dataset_document_filters.keys())
            self.assertIsNotNone(dataset_document_filters[val])

    def test_dataset_document_filters(self):

        set_up_test_sample("test_sample", 200)

        # Test django_lookup_filter
        df = extract_dataset(
            "document_dataset",
            params={
                "outcome_column": "label_id",
                "document_filters": [
                    (
                        "django_lookup_filter",
                        [],
                        {
                            "search_filter": "text__iregex",
                            "search_value": "comedy",
                            "exclude": True,
                        },
                    ),
                    (
                        "django_lookup_filter",
                        [],
                        {
                            "search_filter": "text__iregex",
                            "search_value": "disney",
                            "exclude": False,
                        },
                    ),
                ],
            },
        )
        self.assertEqual(len(df[df["text"].str.contains("[Dd]isney")]), len(df))
        self.assertEqual(len(df[df["text"].str.contains("[Cc]omedy")]), 0)

        # Test filter_by_date
        df = extract_dataset(
            "document_dataset",
            params={
                "outcome_column": "label_id",
                "document_filters": [
                    (
                        "filter_by_date",
                        [],
                        {
                            "min_date": datetime.date(2000, 2, 1),
                            "max_date": datetime.date(2000, 4, 1),
                        },
                    )
                ],
            },
        )
        self.assertEqual(len(df), 49)
        self.assertGreaterEqual(df["date"].min().date(), datetime.date(2000, 2, 1))
        self.assertLessEqual(df["date"].max().date(), datetime.date(2000, 4, 1))

        # Test filter_by_document_ids
        df = extract_dataset(
            "document_dataset",
            params={
                "outcome_column": "label_id",
                "document_filters": [("filter_by_document_ids", [[1, 2, 3, 4, 5]], {})],
            },
        )
        self.assertEqual(
            len(set(df["document_id"].values).intersection(set([1, 2, 3, 4, 5]))),
            len(set(df["document_id"].values)),
        )

        # Test filter_by_other_model_dataset
        model = get_test_model("test")
        good_docs = model.dataset[model.dataset["label_id"] == "10"][
            "document_id"
        ].values
        df = extract_dataset(
            "document_dataset",
            params={
                "outcome_column": "label_id",
                "document_filters": [
                    ("filter_by_other_model_dataset", ["test", "10"], {})
                ],
            },
        )
        self.assertEqual(set(df["document_id"]), set(good_docs))

        # Test filter_by_other_model_prediction
        model.apply_model_to_frame(save=True, refresh=True, num_cores=1)
        df = extract_dataset(
            "document_dataset",
            params={
                "outcome_column": "label_id",
                "document_filters": [
                    ("filter_by_other_model_prediction", ["test", "10"], {})
                ],
            },
        )
        good_docs = model.classifications.filter(label_id="10").values_list(
            "document_id", flat=True
        )
        self.assertEqual(len(set(df["document_id"]).difference(set(good_docs))), 0)

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
