from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project


class ProjectHITTypesTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        pass

    def test_loading(self):

        from django_learning.utils.project_hit_types import project_hit_types

        for val in ["test_hit_type"]:
            self.assertIn(val, project_hit_types.keys())
            self.assertIsNotNone(project_hit_types[val])

    def test_project_hit_types(self):

        from django_learning.models import HITType
        from django_learning.utils.project_hit_types import project_hit_types

        project = Project.objects.create(name="test_project")
        hit_type = HITType.objects.create(name="test_hit_type", project=project)
        config = project_hit_types['test_hit_type']
        for key, value in config.items():
            self.assertEqual(getattr(hit_type, key), value)

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
