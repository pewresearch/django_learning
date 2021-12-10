from __future__ import print_function, absolute_import
from django.conf import settings

from django_learning.models import DocumentClassificationModel, Question

from django_commander.commands import BasicCommand, log_command
from django_pewtils import reset_django_connection
from pewtils.io import FileHandler


class Command(BasicCommand):

    """
    Applies a classification model to the sampling frame it belongs to, creating Classification objects in the database.

    :param name: Name of an existing DocumentClassificationModel
    :param num_cores: (default is 1) number of cores to use
    :param chunk_size: (default is 100000) number of documents to pass to each multiprocessing thread
    :param refresh: (default is False) if True, existing classifications in the database will be recomputed
    """

    parameter_names = ["name"]
    dependencies = []
    test_parameters = {}
    test_options = {}

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("name", type=str)
        parser.add_argument("--num_cores", type=int, default=1)
        parser.add_argument("--chunk_size", type=int, default=10000)
        parser.add_argument("--refresh", action="store_true", default=False)
        return parser

    @log_command
    def run(self):

        DocumentClassificationModel.objects.get(name=self.parameters["name"])
        self.model.extract_dataset(only_load_existing=True)
        self.model.load_model(only_load_existing=True)
        self.model.apply_model_to_frame(
            save=True,
            document_filters=self.model.dataset_extractor.document_filters,
            refresh=self.options["refresh"],
            num_cores=self.options["num_cores"],
            chunk_size=self.options["chunk_size"],
        )
