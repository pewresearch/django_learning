from __future__ import print_function
import unittest
import copy
import time

from django.test import TestCase as DjangoTestCase
from django.conf import settings
from django.core.management import call_command

from django_learning.mturk import MTurk


class BaseTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        pass

    def test_mturk(self):

        import nltk
        from testapp.models import MovieReview
        from django_learning.models import *
        from django_commander.commands import commands

        commands["expire_all_hits_mturk"](sandbox=True).run()
        commands["delete_all_hits_mturk"](sandbox=True).run()

        nltk.download("movie_reviews")

        for fileid in nltk.corpus.movie_reviews.fileids()[:50]:
            document = Document.objects.create(text=nltk.corpus.movie_reviews.raw(fileid))
            review = MovieReview.objects.create(document=document)

        commands["extract_sampling_frame"](
            sampling_frame_name="all_documents"
        ).run()
        commands["create_project"](
            project_name="test_project",
            sandbox=True
        ).run()
        commands["extract_sample"](
            project_name="test_project",
            hit_type_name="test_hit_type",
            sample_name="test_sample",
            sampling_frame_name="all_documents",
            size=10,
            sandbox=True
        ).run()
        commands["create_sample_hits_experts"](
            project_name="test_project",
            sample_name="test_sample",
            num_coders=1,
            sandbox=True
        ).run()
        commands["create_sample_hits_mturk"](
            project_name="test_project",
            sample_name="test_sample",
            num_coders=1,
            sandbox=True
        ).run()
        self.assertTrue(HIT.objects.count() == 20)
        self.assertTrue(HIT.objects.filter(turk=True).count() == 10)
        self.assertTrue(HIT.objects.filter(turk=False).count() == 10)
        print("Go to workersandbox.mturk.com, find the hits, and do a few, then press 'c' to continue")
        print("If the hits don't appear, it's because the MTurk interface can be buggy.")
        print("Check this out instead: https://github.com/jtjacques/mturk-manage")
        print("Also you sometimes need to do more than one assignment for the sandbox API to return anything")
        import pdb
        pdb.set_trace()
        time.sleep(30)
        commands["sync_sample_hits_mturk"](
            project_name="test_project",
            sample_name="test_sample",
            sandbox=True,
            approve=True,
            approve_probability=1.0
        ).run()
        self.assertTrue(Assignment.objects.filter(turk_status="Approved").count()>0)
        for q in Project.objects.get(name="test_project").questions.all():
            self.assertTrue(Code.objects.filter(label__question=q).count()>0)
        HIT.objects.filter(turk_id__isnull=False).filter(assignments__isnull=False).update(turk_id=None)
        commands["expire_sample_hits_mturk"](
            project_name="test_project",
            sample_name="test_sample",
            sandbox=True
        ).run()
        self.assertTrue(HIT.objects.filter(turk_id__isnull=False).count()==10)
        commands["delete_sample_hits_mturk"](
            project_name="test_project",
            sample_name="test_sample",
            sandbox=True
        ).run()
        Project.objects.get(name="test_project", sandbox=True).delete()
        self.assertTrue(HIT.objects.count()==0)
        self.assertTrue(Assignment.objects.count() == 0)
        self.assertTrue(Code.objects.count() == 0)

def tearDown(self):
    pass