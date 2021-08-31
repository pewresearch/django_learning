from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project, set_up_test_sample


class PreprocessorsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        pass

    def test_loading(self):

        from django_learning.utils.preprocessors import preprocessors

        for val in [
            "clean_text",
            "expand_text_cooccurrences",
            # "filter_by_regex",
            "run_function",
        ]:
            self.assertIn(val, preprocessors.keys())
            self.assertIsNotNone(preprocessors[val]())

    def test_clean_text(self):

        from django_learning.utils.preprocessors import preprocessors

        text = "Testing one two three and four but not five. This is an action movie and it's exciting."
        for params, expected in [
            ({}, "testing one two three four five action movie exciting"),
            (
                {"regex_replacers": ["test"]},
                "testing one two three four five action replacer_worked exciting",
            ),
            ({"stopword_sets": ["english"]}, "testing action movie exciting"),
            ({"stopword_sets": ["english", "test"]}, "testing action exciting"),
            ({"regex_filters": ["test"]}, "action movie exciting"),
            (
                {"stopword_sets": ["english"], "stopword_whitelists": ["test"]},
                "testing and action movie and exciting",
            ),
            (
                {"refresh_stopwords": True},
                "testing one two three four five action movie exciting",
            ),
        ]:
            preprocessor = preprocessors["clean_text"](**params)
            result = preprocessor.run(text)
            self.assertEqual(result, expected)

    def test_expand_text_cooccurrences(self):

        from django_learning.utils.preprocessors import preprocessors

        text = "one two three"
        preprocessor = preprocessors["expand_text_cooccurrences"]()
        result = preprocessor.run(text)
        self.assertEqual(result, "one two one three two three")

    def test_run_function(self):

        from django_learning.utils.preprocessors import preprocessors

        text = "one two three"
        preprocessor = preprocessors["run_function"](function=lambda x: "woot")
        result = preprocessor.run(text)
        self.assertEqual(result, "woot")

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
