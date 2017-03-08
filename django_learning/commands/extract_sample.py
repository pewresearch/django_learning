from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample, SamplingFrame, HITType


class Command(BasicCommand):

    parameter_defaults = [
        {"name": "project_name", "type": str, "default": None},
        {"name": "hit_type_name", "type": str, "default": None},
        {"name": "sample_name", "type": str, "default": None}
    ]
    option_defaults = [
        {"name": "sampling_frame_name", "default": "all_documents", "type": str},
        {"name": "sampling_method", "default": "random", "type": str},
        {"name": "size", "default": 0, "type": int},
        {"name": "allow_overlap_with_existing_project_samples", "default": False, "type": bool}
    ]
    dependencies = []

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])
        hit_type = HITType.objects.create_or_update({
            "project": project,
            "name": self.parameters["hit_type_name"]}
        )

        frame, created = SamplingFrame.objects.get_or_create(name=self.options['sampling_frame_name'])
        if frame.documents.count() == 0:
            frame.extract_documents()

        existing = Sample.objects.get_if_exists(
            {"project": project, "name": self.parameters["sample_name"]}
        )
        if existing:
            print "Sample '{}' already exists for project '{}'".format(
                existing,
                project
            )
        else:
            sample = Sample.objects.create_or_update(
                {"project": project, "name": self.parameters["sample_name"]},
                {
                    "hit_type": hit_type,
                    "sampling_method": self.options["sampling_method"],
                    "frame": frame
                }
            )
            sample.extract_documents(
                size=self.options["size"],
                allow_overlap_with_existing_project_samples=self.options["allow_overlap_with_existing_project_samples"]
            )

    def cleanup(self):

        pass