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
            ({"assignments": Assignment.objects.all()[:1]}, 1),
            ({"exclude_coders": Coder.objects.filter(name="coder1")}, 0),
            ({"filter_coders": Coder.objects.filter(name="coder1")}, 25),
            ({"documents": Sample.objects.all()[0].documents.all()[:1]}, 1),
        ]:
            hits = filter_hits(
                project=Project.objects.get(name="test_project"), **params
            )
            self.assertEqual(hits.count(), count)

    def test_filter_assignments(self):

        from django_learning.utils.filters import filter_assignments

        for params, count in [
            ({}, 50),
            ({"sample": Sample.objects.get(name="test_sample")}, 30),
            ({"turk_only": True}, 0),
            ({"experts_only": True}, 50),
            ({"coder_min_hit_count": 50}, 0),
            ({"coder": Coder.objects.get(name="coder1")}, 25),
            ({"completed_only": True}, 50),
            ({"incomplete_only": True}, 0),
            ({"hits": HIT.objects.all()[:1]}, 2),
            ({"exclude_coders": Coder.objects.filter(name="coder1")}, 25),
            ({"filter_coders": Coder.objects.filter(name="coder1")}, 25),
            ({"documents": Sample.objects.all()[0].documents.all()[:1]}, 2),
        ]:
            assignments = filter_assignments(
                project=Project.objects.get(name="test_project"), **params
            )
            self.assertEqual(assignments.count(), count)

    def test_filter_coders(self):

        from django_learning.utils.filters import filter_coders

        hit_ids = list(Coder.objects.get(name="coder1").assignments.values_list("pk", flat=True))
        HIT.objects.filter(pk__in=hit_ids).delete()

        for params, count in [
            ({}, 2),
            ({"sample": Sample.objects.get(name="test_sample")}, 2),
            ({"min_hit_count": 50}, 0),
        ]:
            coders = filter_coders(
                project=Project.objects.get(name="test_project"), **params
            )
            self.assertEqual(coders.count(), count)

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
