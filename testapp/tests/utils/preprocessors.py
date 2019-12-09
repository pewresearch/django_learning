from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project, set_up_test_sample


class PreprocessorsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()
        set_up_test_sample("test_sample", 100)

    def test_preprocessors(self):
        from django_learning.utils.preprocessors import preprocessors

        for val in [
            "clean_text",
            "expand_text_cooccurrences",
            # "filter_by_regex",
            "run_function",
        ]:
            self.assertIn(val, preprocessors.keys())
            self.assertIsNotNone(preprocessors[val]())

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
