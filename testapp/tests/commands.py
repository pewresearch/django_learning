from __future__ import print_function

from django.test import TestCase as DjangoTestCase

from django_learning.models import *


class CommandTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):

        pass

    def test_loading(self):

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
            "django_learning_coding_mturk_exit_sandbox",
            "django_learning_coding_mturk_sync_sample_hits",
            "django_learning_nlp_reload_liwc",
            "django_learning_nlp_reload_nrc_emotions",
            "test_command",
        ]:
            self.assertIn(command, commands.keys())
            params = {p: "1" for p in commands[command].parameter_names}
            self.assertIsNotNone(commands[command](**params))

    def tearDown(self):
        pass
