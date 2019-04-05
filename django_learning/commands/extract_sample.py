from __future__ import print_function
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
        parser.add_argument("--recompute_weights", default=False, action="store_true")
        parser.add_argument("--clear_existing_documents", default=False, action="store_true")
        parser.add_argument("--force_rerun", default=False, action="store_true")
        parser.add_argument("--skip_weighting", default=False, action="store_true")
        parser.add_argument("--sandbox", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"], sandbox=self.options["sandbox"])
        hit_type = HITType.objects.create_or_update({
            "project": project,
            "name": self.parameters["hit_type_name"]
        })

        frame, created = SamplingFrame.objects.get_or_create(name=self.options['sampling_frame_name'])
        if frame.documents.count() == 0:
            frame.extract_documents()

        existing = Sample.objects.get_if_exists(
            {"project": project, "name": self.parameters["sample_name"]}
        )
        if existing and existing.documents.count() > 0 and (not self.options["clear_existing_documents"] and not self.options["recompute_weights"] and not self.options["force_rerun"]):
            print("Sample '{}' already exists for project '{}' (you need to pass --force_rerun, --clear_existing_documents, or --recompute_weights)".format(
                existing,
                project
            ))
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
                allow_overlap_with_existing_project_samples=self.options["allow_overlap_with_existing_project_samples"],
                recompute_weights=self.options["recompute_weights"],
                clear_existing_documents=self.options["clear_existing_documents"],
                skip_weighting=self.options['skip_weighting']
            )

    def cleanup(self):

        pass