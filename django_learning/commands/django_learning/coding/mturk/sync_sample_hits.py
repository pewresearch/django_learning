# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from django_commander.commands import BasicCommand
from django_learning.models import Project, Sample
from django_learning.utils.mturk import MTurk
import time


class Command(BasicCommand):

    """
    Sync with the Mechanical Turk API and download completed HITs. Can optionally approve HITs while looping over
    the completed ones, and can do so slowly over time to make it seem like we're reviewing (to encourage Turkers
    to continue doing a good job).

    :param project_name: Name of an existing project
    :param sample_name: Name of a sample with Mechanical Turk HITs
    :param loop: (default is False) if True, runs indefinitely in a loop to continuously pull completed HITs
    :param time_sleep: (default is 30) how long to sleep between each loop (in seconds)
    :param resync: (default is False) if True, will re-download data for HITs that have already been fully completed
        and synced
    :param approve: (default is False) if True, will approve complete HITs and pay the workers
    :param approve_probability: (default is 1.0) if ``approve=True``, specifies the probability that any given HIT
        will be approved during each loop
    :param update_blocks: (default is False) if True, will update worker blocks; anyone who's been paid the maximum
        compensation amount in the current calendar year (``max_comp``) will be blocked from doing more HITs. This
        really ticks Turkers off, and it's much better to revoke their qualifications instead. Not recommended unless
        you want a bunch of angry Turkers.
    :param max_comp: (default is 500) maximum amount we want to pay a specific Turker in a given year
    :param notify_blocks: (default is False) if True, send Turkers a notification when we block them
    """

    parameter_names = ["project_name", "sample_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("sample_name", type=str)
        parser.add_argument("--time_sleep", default=30, type=int)
        parser.add_argument("--resync", default=False, action="store_true")
        parser.add_argument("--loop", default=False, action="store_true")
        parser.add_argument("--approve", default=False, action="store_true")
        parser.add_argument("--approve_probability", default=1.0, type=float)
        parser.add_argument("--update_blocks", default=False, action="store_true")
        parser.add_argument("--max_comp", default=500, type=int)
        parser.add_argument("--notify_blocks", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"], project=project
        )

        mturk = MTurk(sandbox=project.mturk_sandbox)

        while True:
            mturk.sync_sample_hits(
                sample,
                resync=self.options["resync"],
                approve=self.options["approve"],
                approve_probability=self.options["approve_probability"],
            )
            if self.options["update_blocks"]:
                mturk.update_worker_blocks(
                    notify=self.options["notify_blocks"],
                    max_comp=self.options["max_comp"],
                )
            if not self.options["loop"]:
                break
            time.sleep(self.options["time_sleep"])

    def cleanup(self):

        pass
