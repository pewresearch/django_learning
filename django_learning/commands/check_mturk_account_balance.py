from django_commander.commands import BasicCommand

from django_learning.mturk import MTurk


class Command(BasicCommand):

    parameter_names = []
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        return parser

    def run(self):
        mturk = MTurk(sandbox=False)
        mturk.print_account_balance()