import os

from django.apps import apps as global_apps
from django.db import migrations, models
from django.contrib.contenttypes.models import ContentType

from tqdm import tqdm
from pewtils.django import get_model
from pewtils.io import FileHandler
from django_learning.settings import BASE_DIR


def forwards(apps, schema_editor):

    NgramSet = apps.get_model("django_learning", "NgramSet")
    h = FileHandler(os.path.join(BASE_DIR, "static/django_learning/dictionaries"), use_s3=False)
    df = h.read("NRCv0.92", format="csv", names=["word", "category", "assoc"], header=None, delimiter="\t")
    for cat, words in df.groupby("category"):
        try: nrc_set = NgramSet.objects.get(name=cat, dictionary="nrc_emotions")
        except NgramSet.DoesNotExist: nrc_set = NgramSet(name=cat, dictionary="nrc_emotions")
        nrc_set.label = cat.title()
        nrc_set.words = list(words['word'].values)
        nrc_set.save()
        if cat == "trust":
            # "congressman" is in there, but congresswoman isn't - so I'm adding it
            nrc_set.words.append("congresswoman")
            nrc_set.save()
        print nrc_set

def backwards(apps, schema_editor):

    NgramSet = apps.get_model("django_learning", "NgramSet")
    NgramSet.objects.filter(dictionary="nrc_emotions").delete()


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
