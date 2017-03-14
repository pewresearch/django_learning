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
        pass

    def test_loading(self):
        from django_learning.utils.scoring_functions import scoring_functions

        for val in ["cohens_kappa", "matthews_corrcoef", "maxmin", "mean_difference"]:
            self.assertIn(val, scoring_functions.keys())
            self.assertIsNotNone(scoring_functions[val])

    def test_scoring_functions(self):
        from django_learning.utils.scoring_functions import scoring_functions

        y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        weights = [0.5, 0.5, 1, 1, 1, 1, 1, 1, 1, 1]
        for name, expected, expected_weighted in [
            ("cohens_kappa", 0.62, 0.57),
            ("matthews_corrcoef", 0.67, 0.63),
            ("maxmin", 0.67, 0.6),
            ("mean_difference", 0.67, 0.6),
        ]:
            unweighted = scoring_functions[name](y_true, y_pred, sample_weight=None)
            weighted = scoring_functions[name](y_true, y_pred, sample_weight=weights)
            self.assertAlmostEqual(unweighted, expected, 2)
            self.assertAlmostEqual(weighted, expected_weighted, 2)

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
