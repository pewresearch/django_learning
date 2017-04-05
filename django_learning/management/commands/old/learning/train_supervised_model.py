import pandas, itertools, numpy

from contextlib import closing

from django.core.management.base import BaseCommand, CommandError
from django.db import models

from logos.learning.supervised import DocumentClassificationHandler
from logos.utils.io import FileHandler
from logos.utils import is_null, is_not_null


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("document_type", type=str)
        parser.add_argument("code_variable_name", type=str)
        parser.add_argument("--pipeline", default=None, type=str)
        parser.add_argument("--refresh_training_data", default=False, action="store_true")
        parser.add_argument("--refresh_model", default=False, action="store_true")
        parser.add_argument("--num_cores", default=2, type=int)
        parser.add_argument("--save_to_database", default=None, type=str)

    def handle(self, *args, **options):

        document_type = options.pop("document_type")
        code_variable_name = options.pop("code_variable_name")
        refresh_training_data = options.pop("refresh_training_data")
        refresh_model = options.pop("refresh_model")
        save_name = options.pop("save_to_database")

        if "," in document_type:
            document_type = document_type.split(",")
        h = DocumentClassificationHandler(
            document_type,
            code_variable_name,
            **options
        )
        h.load_training_data(refresh=refresh_training_data)
        h.load_model(refresh=refresh_model)
        h.print_report()
        if save_name:
            h.save_to_database(save_name, compute_cv_scores=True)