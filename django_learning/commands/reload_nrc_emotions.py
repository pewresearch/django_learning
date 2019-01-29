from __future__ import print_function
import os

from django.apps import apps
from pewtils.io import FileHandler

from django_learning.settings import BASE_DIR

from django_commander.commands import BasicCommand


class Command(BasicCommand):
    parameter_names = []
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        return parser

    def run(self):

        NgramSet = apps.get_model("django_learning", "NgramSet")
        h = FileHandler(os.path.join(BASE_DIR, "static/django_learning/dictionaries"), use_s3=False)
        df = h.read("NRCv0.92", format="csv", names=["word", "category", "assoc"], header=None, delimiter="\t")
        for cat, words in df.groupby("category"):
            try:
                nrc_set = NgramSet.objects.get(name=cat, dictionary="nrc_emotions")
            except NgramSet.DoesNotExist:
                nrc_set = NgramSet(name=cat, dictionary="nrc_emotions")
            nrc_set.label = cat.title()
            nrc_set.words = list(words[words['assoc']==1]['word'].values)
            nrc_set.save()
            if cat == "trust":
                # "congressman" is in there, but congresswoman isn't - so I'm adding it
                nrc_set.words.append("congresswoman")
                nrc_set.save()
            print(nrc_set)

    def cleanup(self):

        pass