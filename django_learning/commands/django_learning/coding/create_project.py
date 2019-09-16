from __future__ import print_function
from django_commander.commands import BasicCommand

from django_learning.models import Project


class Command(BasicCommand):

    parameter_names = ["project_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("--sandbox", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.create_or_update({"name": self.parameters["project_name"], "sandbox": self.options["sandbox"]})
        project.save()
        print("Created/updated project '{}'".format(project))

    def cleanup(self):

        pass