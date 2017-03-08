from django.core.management.base import BaseCommand, CommandError
from django.db import models

from django_learning.models import *


class Command(BaseCommand):

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("project_name")
        parser.add_argument("sample_name")

    def handle(self, *args, **options):

        """
        """

        pass
        # TODO: under construction
        # project = Project.objects.get(name=options["project_name"])
        # batch = Batch.objects.get(name=options["batch_name"], project=project)
        #
        # assignments = Assignment.objects\
        #     .filter(hit__batch=batch)\
        #     .exclude(time_started__isnull=True)\
        #     .exclude(time_finished__isnull=True)\
        #     .values("pk", "time_started", "time_finished")
        # df = pandas.DataFrame.from_records(assignments)
        # df["time_delta"] = df.apply(lambda x: (x["time_finished"] - x["time_started"]).seconds, axis=1)

