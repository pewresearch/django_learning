from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample, HIT


class Command(BasicCommand):

    parameter_defaults = [
        {"name": "project_name", "type": str, "default": None},
        {"name": "sample_name", "type": str, "default": None}
    ]
    option_defaults = [
        {"name": "num_coders", "default": 1, "type": int},
        {"name": "template_name", "default": None, "type": str}
    ]
    dependencies = []

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"],
            project=project
        )

        for su in sample.document_units.all():

            HIT.objects.create(
                sample_unit=su,
                turk=False,
                template_name=self.options["template_name"],
                num_coders=self.options["num_coders"]
            )

    def cleanup(self):

        pass