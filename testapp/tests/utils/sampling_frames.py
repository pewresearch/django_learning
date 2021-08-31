from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project


class SamplingFramesTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project(50)

    def test_loading(self):

        from django_learning.utils.sampling_frames import sampling_frames

        for val in ["all_documents", "test"]:
            self.assertIn(val, sampling_frames.keys())
            self.assertIsNotNone(sampling_frames[val]())

    def test_sampling_frames(self):

        from django_learning.models import SamplingFrame

        frame = SamplingFrame.objects.get(name="all_documents")
        self.assertEqual(frame.documents.count(), 50)
        frame = SamplingFrame.objects.create(name="test")
        frame.extract_documents()
        self.assertEqual(frame.documents.count(), 4)

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
