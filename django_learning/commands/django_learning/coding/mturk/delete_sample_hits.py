from __future__ import absolute_import

from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample
from django_learning.utils.mturk import MTurk


class Command(BasicCommand):

    """
    Deletes all of the Mechanical Turk HITs associated with the given sample. Uses the project's ``mturk_sandbox`` flag to
    determine which API to use. It's best to double-check what that's set to before using this command.

    :param project_name: Name of an existing project
    :param sample_name: Name of an exissting sample
    """

    parameter_names = ["project_name", "sample_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("sample_name", type=str)
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"], project=project
        )

        mturk = MTurk(sandbox=project.mturk_sandbox)
        mturk.delete_sample_hits(sample)

    def cleanup(self):

        pass
