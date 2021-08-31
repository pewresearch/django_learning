from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project


class ProjectQualificationScorersTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()

    def test_loading(self):
        from django_learning.utils.project_qualification_scorers import (
            project_qualification_scorers,
        )

        for val in ["test_qualification"]:
            self.assertIn(val, project_qualification_scorers.keys())
            self.assertIsNotNone(project_qualification_scorers[val])

    def test_project_qualification_scorer(self):
        pass

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
