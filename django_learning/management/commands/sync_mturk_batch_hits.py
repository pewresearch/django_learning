import os

from contextlib import closing

from django.core.management.base import BaseCommand, CommandError

from limecoder.models import *
from limecoder.mturk import MTurk
from limecoder.settings import PROJECT_ROOT


class Command(BaseCommand):

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("project_name")
        parser.add_argument("sample_name")
        parser.add_argument("--prod", action="store_true", default=False)
        parser.add_argument("--loop", action="store_true", default=False)

    def handle(self, *args, **options):

        project = Project.objects.get(name=options["project_name"])

        sample = Sample.objects.get(
            name=options["sample_name"],
            project=project
        )

        mturk = MTurk(sandbox=(not options["prod"]))

        # print "Applying latest project inactive_coders"
        # blacklist_path = os.path.join(PROJECT_ROOT, "limecoder", "coders.txt")
        # if os.path.exists(blacklist_path):
        #     with closing(open(blacklist_path, "r")) as input:
        #         for coder_id in input.readlines():
        #             for b in project.batches.all():
        #                 bad_coder = Coder.objects.get(pk=coder_id.strip())
        #                 project.inactive_coders.add(bad_coder)
        #                 mturk.revoke_user_qualification(b, bad_coder)

        if options["loop"]:
            while True:
                mturk.sync_sample_hits(sample)
        else:
            mturk.sync_sample_hits(sample)

