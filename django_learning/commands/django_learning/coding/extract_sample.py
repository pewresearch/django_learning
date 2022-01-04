from __future__ import print_function, absolute_import
from django_commander.commands import BasicCommand

from django_learning.models import Project, Sample, SamplingFrame, HITType


class Command(BasicCommand):

    """
    Extract a sample for an existing project. Can optionally specify the sampling frame, method, and size of the sample.

    :param project_name: Name of an existing project
    :param sample_name: The name to be given to the sample
    :param sampling_frame_name: (default is "all_documents") name of an existing sampling fame
    :param sampling_method: (default is "random") name of an existing sampling method
    :param size: (default is 0, which is equivalent to all documents in the frame) size of the sample to pull
    :param allow_overlap_with_existing_project_samples: (default is False) if True, documents that are already included
        in other samples associated with this project will be eligible to be included in the new sample
    :param recompute_weights: (default is False) if True and the sample already exists, sampling weights will be recomputed
    :param clear_existing_documents: (default is False) if True, if the sample already exists, a fresh sample will be
        pulled and will replace any existing documents currently associated with the sample
    :param seed: (optional) a random seed to use
    :param force_rerun: (default is False) if True, the ``sample.extract_documents`` function will be rerun even if
        the sample already exists. This may cause new documents to be added to the existing sample.
    :param skip_weighting: (default is False) if True, sampling weights will not be computed and cached
    """

    parameter_names = ["project_name", "sample_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("project_name", type=str)
        parser.add_argument("sample_name", type=str)
        parser.add_argument("--sampling_frame_name", default="all_documents", type=str)
        parser.add_argument("--sampling_method", default="random", type=str)
        parser.add_argument("--size", default=0, type=int)
        parser.add_argument(
            "--allow_overlap_with_existing_project_samples",
            default=False,
            action="store_true",
        )
        parser.add_argument("--recompute_weights", default=False, action="store_true")
        parser.add_argument(
            "--clear_existing_documents", default=False, action="store_true"
        )
        parser.add_argument("--seed", default=None, type=int)
        parser.add_argument("--force_rerun", default=False, action="store_true")
        parser.add_argument("--skip_weighting", default=False, action="store_true")
        return parser

    def run(self):

        project = Project.objects.get(name=self.parameters["project_name"])

        frame, created = SamplingFrame.objects.get_or_create(
            name=self.options["sampling_frame_name"]
        )
        if frame.documents.count() == 0:
            frame.extract_documents()

        existing = Sample.objects.get_if_exists(
            {"project": project, "name": self.parameters["sample_name"]}
        )
        if (
            existing
            and existing.documents.count() > 0
            and (
                not self.options["clear_existing_documents"]
                and not self.options["recompute_weights"]
                and not self.options["force_rerun"]
            )
        ):
            print(
                "Sample '{}' already exists for project '{}' (you need to pass --force_rerun, --clear_existing_documents, or --recompute_weights)".format(
                    existing, project
                )
            )
        else:

            sample = Sample.objects.create_or_update(
                {"project": project, "name": self.parameters["sample_name"]},
                {"sampling_method": self.options["sampling_method"], "frame": frame},
            )
            sample.extract_documents(
                size=self.options["size"],
                allow_overlap_with_existing_project_samples=self.options[
                    "allow_overlap_with_existing_project_samples"
                ],
                recompute_weights=self.options["recompute_weights"],
                clear_existing_documents=self.options["clear_existing_documents"],
                skip_weighting=self.options["skip_weighting"],
                seed=self.options["seed"],
            )

    def cleanup(self):

        pass
