from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project


class StopwordSetsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project(10)

    def test_loading(self):

        from django_learning.utils.stopword_sets import stopword_sets
        from django_commander.commands import commands

        commands["django_learning_nlp_extract_entities"](
            document_type="movie_review"
        ).run()
        for val in ["english", "entities", "misc_boilerplate", "months"]:
            self.assertIn(val, stopword_sets.keys())
            stopwords = stopword_sets[val]()
            self.assertIsNotNone(stopwords)
            self.assertGreater(len(stopwords), 0)

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
