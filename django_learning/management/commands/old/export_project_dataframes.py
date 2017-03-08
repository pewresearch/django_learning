# from django.core.management.base import BaseCommand, CommandError
#
# from limecoder.models import *
# from limecoder import dataframes
#
#
# class Command(BaseCommand):
#
#     help = ""
#
#     def add_arguments(self, parser):
#
#         parser.add_argument("project_name")
#         parser.add_argument("--batch_name", type=str)
#         parser.add_argument("--dataframe_name", type=str)
#         parser.add_argument("--refresh", default=False, action="store_true")
#         parser.add_argument("--loop", action="store_true", default=False)
#
#     def handle(self, *args, **options):
#
#         """
#         """
#
#         while True:
#
#             project = Project.objects.get(name=options["project_name"])
#             if options["dataframe_name"]: dataframe_names = [options["dataframe_name"]]
#             else: dataframe_names = dataframes.DATAFRAME_NAMES
#             for df_name in dataframe_names:
#                 df = getattr(dataframes, df_name)
#                 if options["batch_name"]:
#                     b = project.batches.get(name=options["batch_name"])
#                     print "{}, {}, {}".format(project.name, df_name, b.name)
#                     try: df(project, batch=b, refresh=options["refresh"])
#                     except Exception as e: print "Error: {}".format(e)
#                 else:
#                     # print "{}, {}, all batches".format(project.name, df_name)
#                     # try: df(project, refresh=options["refresh"])
#                     # except Exception as e: print "Error: {}".format(e)
#                     # if project.batches.count() > 0:
#                     for b in project.batches.all():
#                         print "{}, {}, {}".format(project.name, df_name, b.name)
#                         try: df(project, batch=b, refresh=options["refresh"])
#                         except Exception as e: print "Error: {}".format(e)
#
#             if not options["loop"]:
#                 break