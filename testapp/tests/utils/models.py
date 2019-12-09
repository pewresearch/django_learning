from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project, set_up_test_sample


class ModelsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()
        set_up_test_sample("test_sample", 100)

    def test_models(self):

        from django_learning.utils.models import models

        # NOTE: in Python 2, sklearn used relative importing in the ensemble module
        # So the decision tree, gradient boosting and random forest models can't import
        for val in [
            # "classification_decision_tree",
            # "classification_gradient_boosting",
            "classification_k_neighbors",
            "classification_linear_svc",
            "classification_multinomial_nb",
            # "classification_random_forest",
            "classification_sgd",
            "classification_svc",
            "classification_xgboost",
            "regression_elastic_net",
            "regression_linear",
            # "regression_random_forest",
            "regression_sgd",
            "regression_svr",
        ]:
            self.assertIn(val, models.keys())
            self.assertIsNotNone(models[val]())

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
