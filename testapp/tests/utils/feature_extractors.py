from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import *


class FeatureExtractorsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()
        set_up_test_sample("test_sample", 100)

    def test_loading(self):
        from django_learning.utils.feature_extractors import feature_extractors

        for val in [
            "django_field_lookups",
            "google_word2vec",
            "ngram_set",
            "preprocessor",
            "punctuation_indicators",
            "regex_counts",
            "tfidf",
            "topics",
            "word2vec",
        ]:
            self.assertIn(val, feature_extractors.keys())
            self.assertIsNotNone(feature_extractors[val]())

    def test_django_field_lookups(self):

        from django_learning.utils.feature_extractors import feature_extractors

        df = extract_dataset("document_dataset")
        extractor = feature_extractors["django_field_lookups"](
            fields=["movie_review__pk"]
        )
        results = extractor.fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertEqual(results["movie_review__pk"].dtype.name, "float64")

    # def test_google_word2vec(self):
    #
    #     from django_learning.utils.feature_extractors import feature_extractors
    #
    #     df = extract_dataset("document_dataset")
    #     extractor = feature_extractors["google_word2vec"](
    #         feature_name_prefix="w2v", limit=100
    #     )
    #     results = extractor.fit_transform(df)
    #     import pdb
    #
    #     pdb.set_trace()
    #     # TODO: finish testing this

    def test_ngram_set(self):

        from django_learning.utils.feature_extractors import feature_extractors
        from django_commander.commands import commands

        commands["django_learning_nlp_reload_liwc"]().run()
        commands["django_learning_nlp_reload_nrc_emotions"]().run()

        df = extract_dataset("document_dataset")

        extractor = feature_extractors["ngram_set"](
            dictionary="liwc",
            ngramset_name=None,
            feature_name_prefix="liwc",
            include_ngrams=False,
        )
        results = extractor.fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertEqual(len(results.columns), 64)

        extractor = feature_extractors["ngram_set"](
            dictionary="liwc",
            ngramset_name="negate",
            feature_name_prefix="liwc",
            include_ngrams=False,
        )
        results = extractor.fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertEqual(len(results.columns), 1)

        extractor = feature_extractors["ngram_set"](
            dictionary="liwc",
            ngramset_name="negate",
            feature_name_prefix="liwc",
            include_ngrams=True,
        )
        results = extractor.fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertEqual(len(results.columns), 58)

        extractor = feature_extractors["ngram_set"](
            dictionary="nrc_emotions",
            ngramset_name=None,
            feature_name_prefix="nrc",
            include_ngrams=False,
        )
        results = extractor.fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertEqual(len(results.columns), 10)

        extractor = feature_extractors["ngram_set"](
            dictionary="nrc_emotions",
            ngramset_name="anger",
            feature_name_prefix="nrc",
            include_ngrams=False,
        )
        results = extractor.fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertEqual(len(results.columns), 1)

        extractor = feature_extractors["ngram_set"](
            dictionary="nrc_emotions",
            ngramset_name="anger",
            feature_name_prefix="nrc",
            include_ngrams=True,
        )
        results = extractor.fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertEqual(len(results.columns), 1248)

    def test_preprocessor(self):

        from django_learning.utils.feature_extractors import feature_extractors

        df = extract_dataset("document_dataset")
        results = feature_extractors["preprocessor"](
            preprocessors=[("run_function", {"function": lambda x: "success"})]
        ).fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertEqual(len(set(results["text"])), 1)
        self.assertEqual(results["text"].values[0], "success")

    def test_punctuation_indicators(self):

        from django_learning.utils.feature_extractors import feature_extractors

        df = extract_dataset("document_dataset")
        results = feature_extractors["punctuation_indicators"](
            feature_name_prefix="punct"
        ).fit_transform(df)
        self.assertEqual(len(results), len(df))
        self.assertGreaterEqual(len(results.columns), 6)

    def test_regex_counts(self):

        from django_learning.utils.feature_extractors import feature_extractors

        df = extract_dataset("document_dataset")
        results = feature_extractors["regex_counts"](regex_filter="test").fit_transform(
            df
        )
        self.assertEqual(len(results), len(df))
        self.assertEqual(len(set(results["text"])), 4)

    def test_tfidf(self):

        from django_learning.utils.feature_extractors import feature_extractors

        df = extract_dataset("document_dataset")
        results = feature_extractors["tfidf"](
            max_df=0.9,
            min_df=2,
            ngram_range=(1, 1),
            preprocessors=[
                (
                    "clean_text",
                    {"process_method": "lemmatize", "stopword_sets": ["english"]},
                )
            ],
        ).fit_transform(df)
        self.assertEqual(results.shape[0], len(df))
        self.assertEqual(results.shape[1], 237)

    def test_topics(self):
        pass
        # TODO: test this

    def test_word2vec(self):
        pass
        # TODO: test this

    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
