from django_commander.commands import BasicCommand

from django_learning.models import Project


class Command(BasicCommand):

    option_defaults = []
    parameter_defaults = [
        {"name": "project_name", "default": None, "type": str}
    ]
    dependencies = []

    def run(self):

        project = Project.objects.create_or_update({"name": self.parameters["project_name"]})
        print project

    def cleanup(self):

        pass