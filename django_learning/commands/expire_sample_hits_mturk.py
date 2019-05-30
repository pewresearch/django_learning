from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample
from django_learning.mturk import MTurk


class Command(BasicCommand):

    parameter_names = ["project_name", "sample_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("sample_name", type=str)
        parser.add_argument("--sandbox", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"], sandbox=self.options["sandbox"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"],
            project=project
        )

        mturk = MTurk(sandbox=self.options["sandbox"])
        mturk.expire_sample_hits(sample)

    def cleanup(self):

        pass