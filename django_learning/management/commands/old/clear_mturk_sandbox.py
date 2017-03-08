import pandas

from django.core.management.base import BaseCommand, CommandError
from django.db import models
from django.template.loader import render_to_string

from boto.mturk.connection import MTurkConnection
from boto.mturk.qualification import Qualifications, PercentAssignmentsApprovedRequirement, NumberHitsApprovedRequirement, LocaleRequirement
from boto.mturk.question import HTMLQuestion

from contextlib import closing

from limecoder.models import *
# from limecoder.settings import AWS_SECRET, AWS_ACCESS
from limecoder.mturk import MTurk


class Command(BaseCommand):

    help = ""

    def handle(self, *args, **options):

        MTurk(sandbox=True).clear_hits()