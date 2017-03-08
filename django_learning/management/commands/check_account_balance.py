from django.core.management.base import BaseCommand, CommandError

from django_learning.models import *
from django_learning.mturk import MTurk


class Command(BaseCommand):

    help = ""

    def handle(self, *args, **options):

        mturk = MTurk(sandbox=False)
        mturk.print_account_balance()