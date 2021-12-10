from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project, set_up_test_sample


class SamplingMethodsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project(50)

    def test_sampling_methods(self):
        from django_learning.utils.sampling_methods import sampling_methods

        for val in ["keyword_oversample"]:
            self.assertIn(val, sampling_methods.keys())
            self.assertIsNotNone(sampling_methods[val]())

        set_up_test_sample("test_sample_keyword_oversample", 4)
        self.assertEqual(
            Sample.objects.get(name="test_sample_keyword_oversample")
            .documents.filter(text__iregex="action")
            .count(),
            2,
        )

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
