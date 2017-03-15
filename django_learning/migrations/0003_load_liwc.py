import os

from django.apps import apps as global_apps
from django.db import migrations, models
from django.contrib.contenttypes.models import ContentType
from StringIO import StringIO
from tqdm import tqdm
from pewtils.django import get_model
from pewtils.io import FileHandler

from django_learning.settings import BASE_DIR


def _get_label(key, val, label=""):
    if type(val) == dict:
        if key in val.keys():
            return "%s > %s" % (label, val[key])
        else:
            for k, v in val.iteritems():
                if label: k = "%s > %s" % (label, k)
                subval = _get_label(key, v, k)
                if subval:
                    return subval
    return ""


def forwards(apps, schema_editor):

    NgramSet = apps.get_model("django_learning", "NgramSet")
    h = FileHandler(os.path.join(BASE_DIR, "static/django_learning/dictionaries"), use_s3=False)

    labels = h.read("liwc_labels", format="json")

    mode = 0
    categories = {}
    words = {}

    liwc_data = h.read("liwc2007", format="dic")

    for line in StringIO(liwc_data):

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

    print "Creating LIWC categories"
    for id, cat in categories.items():
        try: liwc = NgramSet.objects.get(name=cat, dictionary="liwc")
        except NgramSet.DoesNotExist: liwc = NgramSet.objects.create(name=cat, dictionary="liwc")
        categories[id] = liwc

    for word, word_categories in tqdm(words.items(), desc="Loading words"):
        for cat_id in word_categories:
            if word not in categories[cat_id].words:
                categories[cat_id].words.append(word)
                categories[cat_id].save()

    print "Loading labels"
    for cat in categories.values():
        cat.label = _get_label(cat.name, labels)
        cat.save()
        print cat
        valid_categories.append(cat.pk)

    invalid_categories = NgramSet.objects.filter(dictionary="liwc").exclude(pk__in=valid_categories)
    if invalid_categories.count() > 0:
        for invalid_category in tqdm(invalid_categories, desc="Deleting extra/invalid categories",
                                     total=invalid_categories.count()):
            invalid_category.delete()


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
