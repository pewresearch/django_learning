from __future__ import print_function

from django.test import TestCase as DjangoTestCase

from django_learning.models import *


class LoadingTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):

        pass

    def test_config(self):

        from django.conf import settings
        from django_learning.utils.balancing_variables import balancing_variables
        from django_learning.utils.dataset_code_filters import dataset_code_filters
        from django_learning.utils.dataset_coder_filters import dataset_coder_filters
        from django_learning.utils.dataset_document_filters import dataset_document_filters
        from django_learning.utils.dataset_extractors import dataset_extractors
        from django_learning.utils.feature_extractors import feature_extractors
        from django_learning.utils.models import models
        from django_learning.utils.pipelines import pipelines
        from django_learning.utils.preprocessors import preprocessors
        from django_learning.utils.project_hit_types import project_hit_types
        from django_learning.utils.project_qualification_tests import project_qualification_tests
        from django_learning.utils.project_qualification_scorers import project_qualification_scorers
        from django_learning.utils.projects import projects
        from django_learning.utils.regex_filters import regex_filters
        from django_learning.utils.regex_replacers import regex_replacers
        from django_learning.utils.sampling_frames import sampling_frames
        from django_learning.utils.sampling_methods import sampling_methods
        from django_learning.utils.scoring_functions import scoring_functions
        from django_learning.utils.stopword_sets import stopword_sets
        from django_learning.utils.stopword_whitelists import stopword_whitelists
        from django_learning.utils.topic_models import topic_models
        from django_commander.commands import commands

        for val in [
            "document_type",
            "test"
        ]:
            self.assertIn(val, balancing_variables.keys())

        for val in [
            "exclude_by_coder_names",
            "exclude_experts",
            "exclude_mturk",
            "filter_by_coder_names",
            # "filter_by_coder_variance",
            "filter_by_min_coder_doc_count"
        ]:
            self.assertIn(val, dataset_coder_filters.keys())

        for val in [
            "django_lookup_filter",
            "filter_by_date",
            "filter_by_document_ids",
            # "filter_by_existing_code",
            "filter_by_other_model_dataset",
            "filter_by_other_model_prediction",
            "require_all_coders",
            'require_min_coder_count'
        ]:
            self.assertIn(val, dataset_document_filters.keys())

        for val in [
            "document_coder_dataset",
            "document_coder_label_dataset",
            "document_dataset",
            "model_prediction_dataset",
            "raw_document_dataset"
        ]:
            self.assertIn(val, dataset_extractors.keys())

        for val in [
            "django_field_lookups",
            "doc2vec",
            "google_word2vec",
            "ngram_set",
            "preprocessor",
            "punctuation_indicators",
            "regex_counts",
            "tfidf",
            "topics",
            "word2vec"
        ]:
            self.assertIn(val, feature_extractors.keys())

        for val in [
            "classification_decision_tree",
            "classification_gradient_boosting",
            "classification_k_neighbors",
            "classification_linear_svc",
            "classification_multinomial_nb",
            "classification_random_forest",
            "classification_sgd",
            "classification_svc",
            "classification_xgboost",
            "regression_elastic_net",
            "regression_linear",
            "regression_random_forest",
            "regression_sgd",
            "regression_svr"
        ]:
            self.assertIn(val, models.keys())

        for val in [
            "clean_text",
            "expand_text_cooccurrences",
            # "filter_by_regex",
            "run_function"
        ]:
            self.assertIn(val, preprocessors.keys())

        for val in [
            "all_documents"
        ]:
            self.assertIn(val, sampling_frames.keys())

        for val in [
            "cohens_kappa",
            "matthews_corrcoef",
            "maxmin",
            "mean_difference"
        ]:
            self.assertIn(val, scoring_functions.keys())

        for val in [
            "english",
            "entities",
            "misc_boilerplate",
            "months"
        ]:
            self.assertIn(val, stopword_sets.keys())

        for val in [
            "test_hit_type"
        ]:
            self.assertIn(val, project_hit_types.keys())

        for val in [
            "test_qualification"
        ]:
            self.assertIn(val, project_qualification_scorers.keys())

        for val in [
            "test_qualification"
        ]:
            self.assertIn(val, project_qualification_tests.keys())

        for val in [
            "test_project"
        ]:
            self.assertIn(val, projects.keys())

        for command in [
            "check_mturk_account_balance",
            "clear_mturk_sandbox",
            "create_coder",
            "create_project",
            "create_sample_hits_experts",
            "create_sample_hits_mturk",
            "delete_all_hits_mturk",
            "delete_sample_hits_mturk",
            "expire_all_hits_mturk",
            "expire_sample_hits_mturk",
            # "extract_entities",
            'extract_sample',
            'extract_sampling_frame',
            "reload_liwc",
            "reload_nrc_emotions",
            "sync_sample_hits_mturk",
            "test_command"
        ]:
            self.assertIn(command, commands.keys())

    def tearDown(self):
        pass