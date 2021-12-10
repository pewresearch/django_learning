from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project, create_qualification_test


class ProjectQualificationTestsAndScorersTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project(1)

    def test_project_qualification_tests(self):
        from django_learning.utils.project_qualification_tests import (
            project_qualification_tests,
        )
        from django_learning.utils.project_qualification_scorers import (
            project_qualification_scorers,
        )
        from django_learning.models import QualificationTest

        for val in ["test_qualification"]:
            self.assertIn(val, project_qualification_tests.keys())
            self.assertIsNotNone(project_qualification_tests[val])
            self.assertIn(val, project_qualification_scorers.keys())
            self.assertIsNotNone(project_qualification_scorers[val])

        create_qualification_test()
        test = QualificationTest.objects.get(name="test_qualification")
        self.assertTrue(test.is_qualified(Coder.objects.get(name="coder1")))
        self.assertFalse(test.is_qualified(Coder.objects.get(name="coder2")))

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
