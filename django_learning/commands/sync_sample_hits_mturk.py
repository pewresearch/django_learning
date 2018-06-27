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
        parser.add_argument("--time_sleep", default=30, type=int)
        parser.add_argument("--prod", default=False, action="store_true")
        parser.add_argument("--resync", default=False, action="store_true")
        parser.add_argument("--loop", default=False, action="store_true")
        parser.add_argument("--approve_complete_assignments", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"],
            project=project
        )

        mturk = MTurk(sandbox=(not self.options["prod"]))

        while True:
            mturk.sync_sample_hits(sample, resync=self.options["resync"], approve=self.options["approve_complete_assignments"])
            if not self.options["loop"]:
                break
            time.sleep(self.options["time_sleep"])

    def cleanup(self):

        pass
