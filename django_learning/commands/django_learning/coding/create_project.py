from __future__ import print_function
from django_commander.commands import BasicCommand

from django_learning.models import Project


class Command(BasicCommand):

    """
    Create a project based off of a JSON config file. Existing projects will be updated.

    :param project_name: Name of the project to create (must have a corresponding JSON file)
    """

    parameter_names = ["project_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        return parser

    def run(self):

        project = Project.objects.create_or_update(
            {"name": self.parameters["project_name"]}
        )
        project.save()
        print("Created/updated project '{}'".format(project))

    def cleanup(self):

        pass
