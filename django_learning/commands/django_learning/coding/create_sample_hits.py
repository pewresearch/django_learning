from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample, HIT


class Command(BasicCommand):

    parameter_names = ["project_name", "sample_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("sample_name", type=str)
        parser.add_argument("--num_coders", default=1, type=int)
        parser.add_argument("--template_name", default=None, type=str)
        parser.add_argument("--sandbox", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(
            name=self.parameters["project_name"], sandbox=self.options["sandbox"]
        )

        sample = Sample.objects.get(
            name=self.parameters["sample_name"], project=project
        )

        for su in sample.document_units.all():

            HIT.objects.create_or_update(
                {"sample_unit": su, "turk": False},
                {
                    "template_name": self.options["template_name"],
                    "num_coders": self.options["num_coders"],
                },
            )

    def cleanup(self):

        pass
