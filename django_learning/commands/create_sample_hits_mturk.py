from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample
from django_learning.mturk import MTurk


class Command(BasicCommand):

    parameter_defaults = [
        {"name": "project_name", "type": str, "default": None},
        {"name": "sample_name", "type": str, "default": None}
    ]
    option_defaults = [
        {"name": "num_coders", "default": 1, "type": int},
        {"name": "template_name", "default": None, "type": str},
        {"name": "prod", "default": False, "type": bool},
        {"name": "force_hit_type_reset", "default": False, "type": bool},
        {"name": "loop", "default": False, "type": bool}
    ]
    dependencies = []

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        sample = Sample.objects.get(
            name=self.parameters["sample_name"],
            project=project
        )

        mturk = MTurk(sandbox=(not self.options["prod"]))

        if not sample.hit_type.turk_id or self.options["force_hit_type_reset"]:
            mturk.sync_hit_type(sample.hit_type)

        mturk.create_sample_hits(sample, num_coders=self.options["num_coders"], template_name=self.options["template_name"])
        if self.options["loop"]:
            while True:
                mturk.sync_sample_hits(sample)
        else:
            mturk.sync_sample_hits(sample)

        # Create a new hit type with the new code for mturk.create_batch_hit_type(batch)
        # That will also create a new qualification and link it up to the hit type
        # Iterate over all existing HITs and update their HITType to the new one
        # AND iterate over all coders and pending qualification requests, and grant them the new/final hit type's qualification

    def cleanup(self):

        pass