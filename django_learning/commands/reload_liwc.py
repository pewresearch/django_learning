from __future__ import print_function
import os

from django.apps import apps
from django.db import migrations, models
from django.contrib.contenttypes.models import ContentType
from io import StringIO
from tqdm import tqdm
from django_pewtils import get_model
from django.conf import settings
from pewtils.io import FileHandler


from django_commander.commands import BasicCommand

from django_learning.models import SamplingFrame



def _get_label(key, val, label=""):

    if type(val) == dict:
        if key in list(val.keys()):
            return "%s > %s" % (label, val[key])
        else:
            for k, v in val.items():
                if label: k = "%s > %s" % (label, k)
                subval = _get_label(key, v, k)
                if subval:
                    return subval
    return ""


class Command(BasicCommand):

    parameter_names = []
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        return parser

    def run(self):

        NgramSet = apps.get_model("django_learning", "NgramSet")
        h = FileHandler(os.path.join(settings.BASE_DIR, "static/django_learning/dictionaries"), use_s3=False)

        labels = h.read("liwc_labels", format="json")

        mode = 0
        categories = {}
        words = {}

        liwc_data = h.read("liwc2007", format="dic")

        for line in StringIO(unicode(liwc_data)):

            line = line.strip("\r\n")

            if line == "%":
                mode += 1
                continue

            elif mode == 1:
                chunks = line.split("\t")
                categories[chunks[0]] = chunks[1]

            elif mode == 2:
                chunks = line.split("\t")
                word = chunks.pop(0)
                words[word] = chunks

        valid_categories = []

        print("Creating LIWC categories")
        for id, cat in list(categories.items()):
            print("{}, {}".format(id, cat))
            try:
                liwc = NgramSet.objects.get(name=cat, dictionary="liwc")
            except NgramSet.DoesNotExist:
                liwc = NgramSet.objects.create(name=cat, dictionary="liwc")
            categories[id] = liwc
            categories[id].words = []
            categories[id].save()

        for word, word_categories in tqdm(list(words.items()), desc="Loading words"):
            for cat_id in word_categories:
                if word not in categories[cat_id].words:
                    categories[cat_id].words.append(word)
                    categories[cat_id].save()

        print("Loading labels")
        for cat in list(categories.values()):
            cat.label = _get_label(cat.name, labels)
            cat.save()
            print(cat)
            valid_categories.append(cat.pk)

        invalid_categories = NgramSet.objects.filter(dictionary="liwc").exclude(pk__in=valid_categories)
        if invalid_categories.count() > 0:
            for invalid_category in tqdm(invalid_categories, desc="Deleting extra/invalid categories",
                                         total=invalid_categories.count()):
                invalid_category.delete()

    def cleanup(self):

        pass