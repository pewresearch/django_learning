from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project, set_up_test_sample


class FiltersTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()
        set_up_test_sample("test_sample", 15)
        set_up_test_sample("test_sample_holdout", 10)

    def test_filter_hits(self):

        from django_learning.utils.filters import filter_hits

        for params, count in [
            ({}, 25),
            ({"sample": Sample.objects.get(name="test_sample")}, 15),
            ({"turk_only": True}, 0),
            ({"experts_only": True}, 25),
            ({"turk_only": True}, 0),
            ({"finished_only": True}, 25),
            ({"unfinished_only": True}, 0),
            ({"assignments": Assignment.objects.all()[:5]}, 5),
            ({"exclude_coders": Coder.objects.all()[0]}, 0),
            ({"filter_coders": Coder.objects.all()[0]}, 25),
            ({"documents": Sample.objects.all()[0].documents.all()[0]}, 1),
        ]:
            hits = filter_hits(
                project=Project.objects.get(name="test_project"), **params
            )
            if hits.count() != count:
                import pdb

                pdb.set_trace()
            self.assertEqual(hits.count(), count)

        import pdb

        pdb.set_trace()

    # def test_filter_assignments(self):
    #     pass
    #
    # def test_filter_coders(self):
    #     pass

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
