from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project


class ScoringTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()

    def test_scoring_utilities(self):
        pass
        # TODO: test scoring utils
        #   compute_scores_from_datasets_as_coders
        #   compute_scores_from_dataset
        #   get_probability_threshold_score_df
        #   get_probability_threshold_from_score_df
        #   find_probability_threshold
        #   apply_probability_threshold

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
