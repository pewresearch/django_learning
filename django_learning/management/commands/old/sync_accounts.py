# import json, os
#
# from contextlib import closing
#
# from django.core.management.base import BaseCommand, CommandError
# from django.contrib.auth.models import User
#
# from limecoder.settings import PROJECT_ROOT
# from limecoder.models import Coder
#
#
# class Command(BaseCommand):
#
#     help = ""
#
#     def handle(self, *args, **options):
#
#         with closing(open(os.path.join(PROJECT_ROOT, "limecoder", "coders.json"), "r")) as input:
#             coders = json.load(input)
#         for c in coders:
#             try:
#                 user = User.objects.create_user(c['username'], c['email'], c['password'])
#             except:
#                 user = User.objects.get(username=c['username'])
#                 user.set_password(c['password'])
#                 user.email = c['email']
#                 user.save()
#             Coder.objects.create_or_update({"user": user}, {"name": c['username']}, return_object=False)
#
#             print user
#
#
