from __future__ import print_function
import unittest
import copy

from django.test import TestCase as DjangoTestCase
from django.conf import settings
from django.core.management import call_command

class BaseTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        pass

    def test_mturk(self):

        import nltk
        from testapp.models import MovieReview
        from django_learning.models import Document
        from django_commander.commands import commands

        nltk.download("movie_reviews")

        for fileid in nltk.corpus.movie_reviews.fileids()[:50]:
            document = Document.objects.create(text=nltk.corpus.movie_reviews.raw(fileid))
            review = MovieReview.objects.create(document=document)

        commands["create_project"](project_name="test_project").run()
        commands["extract_sampling_frame"](sampling_frame_name="all_documents").run()
        commands["extract_sample"](
            project_name="test_project",
            hit_type_name="test_hit_type",
            sample_name="test_sample",
            sampling_frame_name="all_documents",
            size=10
        ).run()
        commands["create_sample_hits_experts"](
            project_name="test_project",
            sample_name="test_sample",
            num_coders=1
        ).run()
        commands["create_sample_hits_mturk"](
            project_name="test_project",
            sample_name="test_sample",
            num_coders=1
        ).run()
        commands["sync_sample_hits_mturk"](
            project_name="test_project",
            sample_name="test_sample"
        ).run()
        commands["cancel_sample_hits_mturk"](
            project_name="test_project",
            sample_name="test_sample"
        ).run()
        commands["clear_mturk_sandbox"]().run()

def tearDown(self):
        pass