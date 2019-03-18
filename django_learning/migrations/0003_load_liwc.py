
import os

from django.apps import apps as global_apps
from django.db import migrations, models
from django.contrib.contenttypes.models import ContentType
from io import StringIO
from tqdm import tqdm
from django_pewtils import get_model
from pewtils.io import FileHandler

from django_learning.settings import BASE_DIR
from django_commander.commands import commands


def forwards(apps, schema_editor):

    commands["reload_liwc"]().run()

def backwards(apps, schema_editor):

    NgramSet = apps.get_model("django_learning", "NgramSet")
    NgramSet.objects.filter(dictionary="liwc").delete()


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0002_load_nrc_emotions'),
    ]

    operations = [
        migrations.RunPython(forwards, backwards),
    ]
