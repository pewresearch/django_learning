from django.core.management.base import BaseCommand, CommandError

from django_learning.models import *
from django_learning.mturk import MTurk


class Command(BaseCommand):

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("project_name")
        parser.add_argument("sample_name")
        parser.add_argument("--prod", action="store_true", default=False)

    def handle(self, *args, **options):

        sample = Sample.objects.get(
            name=options["batch_name"],
            project=Project.objects.get(name=options["project_name"])
        )

        mturk = MTurk(sandbox=(not options["prod"]))

        for b in sample.hits.filter(turk_id__isnull=False):
            mturk.conn.expire_hit(b.turk_id)