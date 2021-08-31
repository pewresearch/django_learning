from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import *


class DatasetCodeFiltersTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        pass

    def test_loading(self):
        from django_learning.utils.dataset_code_filters import dataset_code_filters

        for val in ["test"]:
            self.assertIn(val, dataset_code_filters.keys())
            self.assertIsNotNone(dataset_code_filters[val])

    def test_dataset_code_filters(self):

        set_up_test_sample("test_sample", 20)

        # Test without code filter
        df = extract_dataset("document_coder_label_dataset")
        self.assertEqual(len(df), 40)

        # Test with code filter
        df = extract_dataset(
            "document_coder_label_dataset", params={"code_filters": [("test", [], {})]}
        )
        pk = Question.objects.get(name="test_checkbox").labels.get(value="1").pk
        self.assertEqual(len(df), 2)
        self.assertEqual(df["label_id"].nunique(), 1)
        self.assertEqual(df["label_id"].values[0], pk)

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
