# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from django_commander.commands import BasicCommand
from django_learning.models import Project, Sample
from django_learning.mturk import MTurk
import time


class Command(BasicCommand):

    parameter_names = ["project_name", "sample_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("sample_name", type=str)
        parser.add_argument("--num_coders", default=1, type=int)
        parser.add_argument("--template_name", default=None, type=str)
        parser.add_argument("--prod", default=False, action="store_true")
        parser.add_argument("--force_hit_type_reset", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"],
            project=project
        )

        mturk = MTurk(sandbox=(not self.options["prod"]))

        # TODO: this can become a problem when you create a hit type in prod, but then on a new sample you want to try sandbox again
        # the project already has an API ID for the hit type (the correct prod one, which may be associated with existing Turker qualification test results)
        # but if you switch back to the sandbox, the API won't recognize the prod ID
        # easiest solution is to modify the HitType model to have a prod and non-prod ID
        if not sample.hit_type.turk_id or self.options["force_hit_type_reset"]:
            mturk.sync_hit_type(sample.hit_type)

        mturk.create_sample_hits(sample, num_coders=self.options["num_coders"], template_name=self.options["template_name"])

        # Create a new hit type with the new code for mturk.create_batch_hit_type(batch)
        # That will also create a new qualification and link it up to the hit type
        # Iterate over all existing HITs and update their HITType to the new one
        # AND iterate over all coders and pending qualification requests, and grant them the new/final hit type's qualification

    def cleanup(self):

        pass
