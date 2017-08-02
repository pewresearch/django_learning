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
        parser.add_argument("--prod", default=False, action="store_true")
        parser.add_argument("--loop", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"],
            project=project
        )

        mturk = MTurk(sandbox=(not self.options["prod"]))

        if self.options["loop"]:
            while True:
                mturk.sync_sample_hits(sample)
        else:
            mturk.sync_sample_hits(sample)

    def cleanup(self):

        pass