# from django.core.management.base import BaseCommand, CommandError
#
# from democracy.models import *
#
#
# class Command(BaseCommand):
#
#     help = ""
#
#     def add_arguments(self, parser):
#
#         parser.add_argument("name")
#
#     def handle(self, *args, **options):
#
#         TopicModel.objects.get(name=options["name"]).delete()
#         print "Model '%s' deleted" % options["name"]