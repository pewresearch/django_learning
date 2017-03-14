# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from django_commander.commands import BasicCommand
from django_learning.models import Project, Sample, HITType
from django_learning.utils.mturk import MTurk


class Command(BasicCommand):
    """
    Create and launch Mechanical Turk HITs for an existing project and sample. Uses the project's ``mturk_sandbox`` flag
    to determine whether or not to use the sandbox.

    :param project_name: Name of an existing project
    :param sample_name: Name of an existing sample
    :param hit_type_name: Name of an existing HIT type to assign to the newly created HITs
    :param num_coders: (default is 1) number of coders to complete each HIT
    :param template_name: (optional) the name of a custom project_hit_template
    :param force_hit_type_reset: (default is False) if True, the HIT type will be reset via the MTurk API, which
        may be necessary - sometimes the API screws up, but this fixes it up
    """

    parameter_names = ["project_name", "sample_name", "hit_type_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("sample_name", type=str)
        parser.add_argument("hit_type_name", type=str)
        parser.add_argument("--num_coders", default=1, type=int)
        parser.add_argument("--template_name", default=None, type=str)
        parser.add_argument(
            "--force_hit_type_reset", default=False, action="store_true"
        )
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"], project=project
        )

        hit_type = HITType.objects.create_or_update(
            {"project": project, "name": self.parameters["hit_type_name"]}
        )

        mturk = MTurk(sandbox=project.mturk_sandbox)
        if not hit_type.turk_id or self.options["force_hit_type_reset"]:
            mturk.sync_hit_type(hit_type)

        mturk.create_sample_hits(
            sample,
            hit_type,
            num_coders=self.options["num_coders"],
            template_name=self.options["template_name"],
        )

        # Create a new hit type with the new code for mturk.create_batch_hit_type(batch)
        # That will also create a new qualification and link it up to the hit type
        # Iterate over all existing HITs and update their HITType to the new one
        # AND iterate over all coders and pending qualification requests, and grant them the new/final hit type's qualification

    def cleanup(self):

        pass
