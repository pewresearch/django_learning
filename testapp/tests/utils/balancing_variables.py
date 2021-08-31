from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import *


class BalancingVariablesTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        pass

    def test_loading(self):

        from django_learning.utils.balancing_variables import balancing_variables

        for val in ["document_type", "test"]:
            self.assertIn(val, balancing_variables.keys())
            self.assertIsNotNone(balancing_variables[val])

    def test_balancing_variables(self):

        set_up_test_sample("test_sample", 100)

        # Test without balancing variable
        df = extract_dataset("document_dataset")
        self.assertNotIn("balancing_weight", df.columns)

        # Test with balancing variable
        df = extract_dataset(
            "document_dataset", params={"balancing_variables": ["test"]}
        )
        self.assertEqual(df["balancing_weight"].nunique(), 4)
        self.assertAlmostEqual(df["balancing_weight"].min(), 0.87, 2)
        self.assertAlmostEqual(df["balancing_weight"].max(), 1.11, 2)

        # Test to make sure model properly uses balancing variable
        model = get_test_model("test_with_balancing_variables", run=False)
        model.extract_dataset(refresh=True)
        self.assertEqual(
            model.dataset["training_weight"].min(), df["balancing_weight"].min()
        )
        self.assertEqual(
            model.dataset["training_weight"].max(), df["balancing_weight"].max()
        )

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
