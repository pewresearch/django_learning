from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project


class StopwordWhitelistsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()

    def test_stopword_whitelists(self):
        from django_learning.utils.stopword_whitelists import stopword_whitelists

        for val in ["test"]:
            self.assertIn(val, stopword_whitelists.keys())
            self.assertIsNotNone(stopword_whitelists[val]())

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
