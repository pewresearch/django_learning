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
        parser.add_argument("--sandbox", default=False, action="store_true")
        parser.add_argument("--resync", default=False, action="store_true")
        parser.add_argument("--loop", default=False, action="store_true")
        parser.add_argument("--approve", default=False, action="store_true")
        parser.add_argument("--approve_probability", default=1.0, type=float)
        parser.add_argument("--update_blocks", default=False, action="store_true")
        parser.add_argument("--max_comp", default=500, type=int)
        parser.add_argument("--notify_blocks", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"], sandbox=self.options["sandbox"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"],
            project=project
        )

        mturk = MTurk(sandbox=self.options["sandbox"])

        while True:
            mturk.sync_sample_hits(
                sample,
                resync=self.options["resync"],
                approve=self.options["approve"],
                approve_probability=self.options["approve_probability"]
            )
            if self.options["update_blocks"]:
                mturk.update_worker_blocks(
                    notify=self.options["notify_blocks"],
                    max_comp=self.options["max_comp"]
                )
            if not self.options["loop"]:
                break
            time.sleep(self.options["time_sleep"])

    def cleanup(self):

        pass
