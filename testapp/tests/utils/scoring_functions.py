from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project


class ScoringFunctionsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()

    def test_scoring_functions(self):
        from django_learning.utils.scoring_functions import scoring_functions

        for val in ["cohens_kappa", "matthews_corrcoef", "maxmin", "mean_difference"]:
            self.assertIn(val, scoring_functions.keys())
            self.assertIsNotNone(scoring_functions[val])

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
