from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample, SamplingFrame, HITType


class Command(BasicCommand):

    parameter_names = ["project_name", "hit_type_name", "sample_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("hit_type_name", type=str)
        parser.add_argument("sample_name", type=str)
        parser.add_argument("--sampling_frame_name", default="all_documents", type=str)
        parser.add_argument("--sampling_method", default="random", type=str)
        parser.add_argument("--size", default=0, type=int)
        parser.add_argument("--allow_overlap_with_existing_project_samples", default=False, action="store_true")
        return parser

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