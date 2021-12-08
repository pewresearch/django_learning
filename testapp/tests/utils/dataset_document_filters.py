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
            # "filter_by_existing_code",
            "filter_by_other_model_dataset",
            "filter_by_other_model_prediction",
        ]:
            self.assertIn(val, dataset_document_filters.keys())
            self.assertIsNotNone(dataset_document_filters[val])

    def test_dataset_document_filters(self):

        set_up_test_sample("test_sample", 20)

        # Test django_lookup_filter
        df = extract_dataset(
            "document_coder_label_dataset",
            params={
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
                ]
            },
        )
        self.assertEqual(len(df[df["text"].str.contains("[Dd]isney")]), len(df))
        self.assertEqual(len(df[df["text"].str.contains("[Cc]omedy")]), 0)

        # Test filter_by_date
        df = extract_dataset(
            "document_coder_label_dataset",
            params={
                "document_filters": [
                    (
                        "filter_by_date",
                        [],
                        {
                            "min_date": datetime.date(2000, 2, 1),
                            "max_date": datetime.date(2000, 4, 1),
                        },
                    )
                ]
            },
        )
        self.assertEqual(len(df), 22)
        self.assertGreaterEqual(df["date"].min().date(), datetime.date(2000, 2, 1))
        self.assertLessEqual(df["date"].max().date(), datetime.date(2000, 4, 1))

        # Test filter_by_document_ids
        df = extract_dataset(
            "document_coder_label_dataset",
            params={
                "document_filters": [("filter_by_document_ids", [[1, 2, 3, 4, 5]], {})]
            },
        )
        self.assertEqual(
            len(set(df["document_id"].values).intersection(set([1, 2, 3, 4, 5]))),
            len(set(df["document_id"].values)),
        )

        # TODO: test filter_by_other_model_dataset
        # TODO: test filter_by_other_model_prediction
        # TODO: test_require_all_coders
        # TODO: test_require_min_coder_count

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
