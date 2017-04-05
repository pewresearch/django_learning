# from django.core.management.base import BaseCommand, CommandError
#
# from democracy.models import CodeVariable
#
#
# class Command(BaseCommand):
#     """
#     """
#     help = ""
#
#     def add_arguments(self, parser):
#
#         parser.add_argument("--code_variable", default=None, type=str)
#
#     def handle(self, *args, **options):
#
#         if options["code_variable"] == "all":
#             code_vars = CodeVariable.objects.all()
#         else:
#             code_vars = [CodeVariable.objects.get(name=options["code_variable"])]
#
#         for code_variable in code_vars:
#
#             if code_variable.model:
#
#                 code_variable.model.print_top_features()
