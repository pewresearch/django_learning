from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import *


class DatasetCoderFiltersTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        pass

    def test_loading(self):
        from django_learning.utils.dataset_coder_filters import dataset_coder_filters

        for val in [
            "exclude_by_coder_names",
            "exclude_experts",
            "exclude_mturk",
            "filter_by_coder_names",
            # "filter_by_coder_variance",
            "filter_by_min_coder_doc_count",
        ]:
            self.assertIn(val, dataset_coder_filters.keys())
            self.assertIsNotNone(dataset_coder_filters[val])

    def test_dataset_coder_filters(self):

        set_up_test_sample("test_sample", 20)

        # Test without code filter
        df = extract_dataset("document_coder_label_dataset")
        self.assertEqual(len(df), 40)

        # Test exclude_by_coder_names
        df = extract_dataset(
            "document_coder_label_dataset",
            params={"coder_filters": [("exclude_by_coder_names", [["coder1"]], {})]},
        )
        self.assertEqual(len(df), 20)
        self.assertEqual(df["coder_name"].nunique(), 1)
        self.assertEqual(df["coder_name"].values[0], "coder2")

        # Test exclude_experts
        df = extract_dataset(
            "document_coder_label_dataset",
            params={"coder_filters": [("exclude_experts", [], {})]},
        )
        self.assertEqual(len(df), 0)

        # Test exclude_mturk
        df = extract_dataset(
            "document_coder_label_dataset",
            params={"coder_filters": [("exclude_mturk", [], {})]},
        )
        self.assertEqual(len(df), 40)

        # Test filter_by_coder_names
        df = extract_dataset(
            "document_coder_label_dataset",
            params={"coder_filters": [("filter_by_coder_names", [["coder1"]], {})]},
        )
        self.assertEqual(len(df), 20)
        self.assertEqual(df["coder_name"].nunique(), 1)
        self.assertEqual(df["coder_name"].values[0], "coder1")

        # Test filter_by_min_coder_doc_count
        assignments = Assignment.objects.filter(coder__name="coder1")[:10]
        Assignment.objects.filter(pk__in=assignments).delete()
        df = extract_dataset(
            "document_coder_label_dataset",
            params={
                "coder_filters": [
                    ("filter_by_min_coder_doc_count", [], {"min_docs": 15})
                ]
            },
        )
        self.assertEqual(len(df), 20)
        self.assertEqual(df["coder_name"].nunique(), 1)
        self.assertEqual(df["coder_name"].values[0], "coder2")

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
