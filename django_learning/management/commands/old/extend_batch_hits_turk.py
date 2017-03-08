# from django.core.management.base import BaseCommand, CommandError
#
# from limecoder.models import *
# from limecoder.mturk import MTurk
#
#
# class Command(BaseCommand):
#
#     help = ""
#
#     def add_arguments(self, parser):
#
#         parser.add_argument("project_name")
#         parser.add_argument("batch_name")
#         parser.add_argument("--add_days", type=int, default=None)
#         parser.add_argument("--add_coders", type=int, default=None)
#         parser.add_argument("--prod", action="store_true", default=False)
#
#     def handle(self, *args, **options):
#
#         batch = Batch.objects.get(
#             name=options["batch_name"],
#             project=Project.objects.get(name=options["project_name"])
#         )
#
#         mturk = MTurk(sandbox=(not options["prod"]))
#
#         if options["add_days"]: days = 60*60*24*options["add_days"]
#         else: days = None
#
#         for b in batch.hits.filter(turk_id__isnull=False):
#
#             mturk.conn.extend_hit(b.turk_id, assignments_increment=options.get("add_coders", None), expiration_increment=days)