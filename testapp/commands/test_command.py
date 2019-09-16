from __future__ import print_function

from django_commander.commands import BasicCommand, log_command


class Command(BasicCommand):

    """
    """

    parameter_names = []
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        return parser

    def __init__(self, **options):

        super(Command, self).__init__(**options)

    @log_command
    def run(self):

        pass
