# import subprocess
#
# from django.core.management.base import BaseCommand, CommandError
# from django.db import connection
#
#
# class Command(BaseCommand):
#
#     help = "Drops the ENTIRE database and recreates it fresh"
#
#     def handle(self, *args, **options):
#
#         sql_pipe = subprocess.Popen(["python", "manage.py", "sqlflush"], stdout=subprocess.PIPE)
#         output = subprocess.check_output(["python", "manage.py", "dbshell"], stdin=sql_pipe.stdout)
#         print output
#
#         with connection.cursor() as cursor:
#             cursor.execute("drop schema public cascade")
#             cursor.execute("create schema public")
#
#         print "Success!  Database is completely fresh now."