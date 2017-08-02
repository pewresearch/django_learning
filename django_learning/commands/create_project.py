from django_commander.commands import BasicCommand

from django_learning.models import Project


class Command(BasicCommand):

    parameter_names = ["project_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        return parser

    def run(self):

        project = Project.objects.create_or_update({"name": self.parameters["project_name"]})
        project.save()
        print "Created/updated project '{}'".format(project)

    def cleanup(self):

        pass