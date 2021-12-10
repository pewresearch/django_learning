from __future__ import absolute_import

from django_commander.commands import BasicCommand

from django_learning.utils.mturk import MTurk


class Command(BasicCommand):

    parameter_names = ["project_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])
        project.mturk_sandbox = True
        project.save()

    def cleanup(self):

        pass
