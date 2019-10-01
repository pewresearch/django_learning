from __future__ import absolute_import

from django_commander.commands import BasicCommand

from django_learning.utils.mturk import MTurk


class Command(BasicCommand):

    parameter_names = []
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("--sandbox", default=False, action="store_true")
        return parser

    def run(self):

        mturk = MTurk(sandbox=self.options["sandbox"])
        mturk.expire_all_hits()

    def cleanup(self):

        pass
