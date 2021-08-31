from __future__ import absolute_import

from django_commander.commands import BasicCommand
from pewtils import is_not_null
from django_learning.models import DocumentClassificationModel, Sample, SamplingFrame


class Command(BasicCommand):

    parameter_names = ["name", "pipeline_name"]
    dependencies = []
    test_parameters = {}
    test_options = {}

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("name", type=str)
        parser.add_argument("pipeline_name", type=str)
        parser.add_argument("--refresh_dataset", action="store_true", default=False)
        parser.add_argument("--refresh_model", action="store_true", default=False)
        parser.add_argument("--num_cores", type=int, default=2)
        return parser

    def run(self):

        from django_learning.utils.pipelines import pipelines

        pipeline = pipelines[self.parameters["pipeline_name"]]()
        sample_names = pipeline["dataset_extractor"]["parameters"]["sample_names"]
        if "test_dataset_extractor" in pipeline.keys():
            sample_names.extend(
                pipeline["test_dataset_extractor"]["parameters"]["sample_names"]
            )
        samples = Sample.objects.filter(name__in=sample_names)
        frames = SamplingFrame.objects.filter(samples__in=samples).distinct()
        if frames.count() == 1:
            DocumentClassificationModel.objects.filter(
                name=self.parameters["name"]
            )  # .update(parameters={})
            model = DocumentClassificationModel.objects.create_or_update(
                {
                    "name": self.parameters["name"],
                    "pipeline_name": self.parameters["pipeline_name"],
                },
                {"sampling_frame": frames[0]},
            )
            model.extract_dataset(refresh=self.options["refresh_dataset"])
            model.load_model(
                refresh=(
                    self.options["refresh_model"] or self.options["refresh_dataset"]
                ),
                num_cores=self.options["num_cores"],
                only_load_existing=self.options["export_results"],
            )
            if is_not_null(model.model):
                model.describe_model()

                if self.options["refresh_dataset"] or self.options["refresh_model"]:
                    model.get_cv_prediction_results(refresh=True)
                    model.get_test_prediction_results(refresh=True)
                    model.find_probability_threshold(save=True)
                test_scores = model.get_test_prediction_results()
                fold_scores = model.get_cv_prediction_results()
        else:
            raise Exception(
                "The dataset extractor specified in the pipeline uses samples from multiple frames"
            )
