from __future__ import print_function

from django.test import TestCase as DjangoTestCase

from django_learning.models import *


class LoadingTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):

        pass

    def test_balancing_variables(self):

        from django_learning.utils.balancing_variables import balancing_variables

        for val in ["document_type", "test"]:
            self.assertIn(val, balancing_variables.keys())
            self.assertIsNotNone(balancing_variables[val])

    def test_dataset_coder_filters(self):

        from django_learning.utils.dataset_coder_filters import dataset_coder_filters

        for val in [
            "exclude_by_coder_names",
            "exclude_experts",
            "exclude_mturk",
            "filter_by_coder_names",
            # "filter_by_coder_variance",
            "filter_by_min_coder_doc_count",
        ]:
            self.assertIn(val, dataset_coder_filters.keys())
            self.assertIsNotNone(dataset_coder_filters[val])

    def test_dataset_code_filters(self):

        from django_learning.utils.dataset_code_filters import dataset_code_filters

        for val in ["test"]:
            self.assertIn(val, dataset_code_filters.keys())
            self.assertIsNotNone(dataset_code_filters[val])

    def test_dataset_document_filters(self):

        from django_learning.utils.dataset_document_filters import (
            dataset_document_filters,
        )

        for val in [
            "django_lookup_filter",
            "filter_by_date",
            "filter_by_document_ids",
            # "filter_by_existing_code",
            "filter_by_other_model_dataset",
            "filter_by_other_model_prediction",
            "require_all_coders",
            "require_min_coder_count",
        ]:
            self.assertIn(val, dataset_document_filters.keys())
            self.assertIsNotNone(dataset_document_filters[val])

    def test_dataset_extractors(self):

        from django_learning.utils.dataset_extractors import dataset_extractors

        for val in [
            "document_coder_dataset",
            "document_coder_label_dataset",
            "document_dataset",
            "model_prediction_dataset",
            "raw_document_dataset",
        ]:
            self.assertIn(val, dataset_extractors.keys())
            self.assertIsNotNone(dataset_extractors[val])

    def test_feature_extractors(self):

        from django_learning.utils.feature_extractors import feature_extractors

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
            "word2vec",
        ]:
            self.assertIn(val, feature_extractors.keys())
            self.assertIsNotNone(feature_extractors[val]())

    def test_models(self):

        from django_learning.utils.models import models

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
            "regression_svr",
        ]:
            self.assertIn(val, models.keys())
            self.assertIsNotNone(models[val]())

    def test_preprocessors(self):

        from django_learning.utils.preprocessors import preprocessors

        for val in [
            "clean_text",
            "expand_text_cooccurrences",
            # "filter_by_regex",
            "run_function",
        ]:
            self.assertIn(val, preprocessors.keys())
            self.assertIsNotNone(preprocessors[val]())

    def test_sampling_frames(self):

        from django_learning.utils.sampling_frames import sampling_frames

        for val in ["all_documents"]:
            self.assertIn(val, sampling_frames.keys())
            self.assertIsNotNone(sampling_frames[val]())

    def test_scoring_functions(self):

        from django_learning.utils.scoring_functions import scoring_functions

        for val in ["cohens_kappa", "matthews_corrcoef", "maxmin", "mean_difference"]:
            self.assertIn(val, scoring_functions.keys())
            self.assertIsNotNone(scoring_functions[val])

    def test_stopword_sets(self):

        from django_learning.utils.stopword_sets import stopword_sets

        for val in ["english", "entities", "misc_boilerplate", "months"]:
            self.assertIn(val, stopword_sets.keys())
            self.assertIsNotNone(stopword_sets[val]())

    def test_project_hit_types(self):

        from django_learning.utils.project_hit_types import project_hit_types

        for val in ["test_hit_type"]:
            self.assertIn(val, project_hit_types.keys())
            self.assertIsNotNone(project_hit_types[val])

    def test_project_qualification_scorers(self):

        from django_learning.utils.project_qualification_scorers import (
            project_qualification_scorers,
        )

        for val in ["test_qualification"]:
            self.assertIn(val, project_qualification_scorers.keys())
            self.assertIsNotNone(project_qualification_scorers[val])

    def test_project_qualification_tests(self):

        from django_learning.utils.project_qualification_tests import (
            project_qualification_tests,
        )

        for val in ["test_qualification"]:
            self.assertIn(val, project_qualification_tests.keys())
            self.assertIsNotNone(project_qualification_tests[val])

    def test_projects(self):

        from django_learning.utils.projects import projects

        for val in ["test_project"]:
            self.assertIn(val, projects.keys())
            self.assertIsNotNone(projects[val])

    def test_pipelines(self):

        from django_learning.utils.pipelines import pipelines

        for val in ["test", "test_with_holdout"]:
            self.assertIn(val, pipelines.keys())
            self.assertIsNotNone(pipelines[val])

    def test_regex_filters(self):

        from django_learning.utils.regex_filters import regex_filters

        for val in ["test"]:
            self.assertIn(val, regex_filters.keys())
            self.assertIsNotNone(regex_filters[val]())

    def test_regex_replacers(self):

        from django_learning.utils.regex_replacers import regex_replacers

        for val in ["test"]:
            self.assertIn(val, regex_replacers.keys())
            self.assertIsNotNone(regex_replacers[val]())

    def test_sampling_methods(self):

        from django_learning.utils.sampling_methods import sampling_methods

        for val in ["test"]:
            self.assertIn(val, sampling_methods.keys())
            self.assertIsNotNone(sampling_methods[val]())

    def test_stopword_whitelists(self):

        from django_learning.utils.stopword_whitelists import stopword_whitelists

        for val in ["test"]:
            self.assertIn(val, stopword_whitelists.keys())
            self.assertIsNotNone(stopword_whitelists[val]())

    def test_topic_models(self):

        from django_learning.utils.topic_models import topic_models

        for val in ["test"]:
            self.assertIn(val, topic_models.keys())
            self.assertIsNotNone(topic_models[val]())

    def test_commands(self):

        from django_commander.commands import commands

        for command in [
            "django_learning_coding_create_coder",
            "django_learning_coding_create_project",
            "django_learning_coding_create_sample_hits",
            "django_learning_coding_extract_sample",
            "django_learning_coding_extract_sampling_frame",
            "django_learning_coding_mturk_create_sample_hits",
            "django_learning_coding_mturk_delete_all_hits",
            "django_learning_coding_mturk_delete_sample_hits",
            "django_learning_coding_mturk_expire_all_hits",
            "django_learning_coding_mturk_expire_sample_hits",
            "django_learning_coding_mturk_check_account_balance",
            "django_learning_coding_mturk_clear_sandbox",
            "django_learning_coding_mturk_sync_sample_hits",
            # "django_learning_nlp_extract_entities",
            "django_learning_nlp_reload_liwc",
            "django_learning_nlp_reload_nrc_emotions",
            "test_command",
        ]:
            self.assertIn(command, commands.keys())
            params = {p: "1" for p in commands[command].parameter_names}
            self.assertIsNotNone(commands[command](**params))

    def tearDown(self):
        pass
