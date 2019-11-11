# -*- coding: utf-8 -*-
# Generated by Django 1.11.26 on 2019-11-11 10:13
from __future__ import unicode_literals

from django.db import migrations, models


def forwards(apps, schema_editor):
    HIT = apps.get_model("django_learning", "HIT")
    for hit in HIT.objects.all():
        hit.save()

def backwards(apps, schema_editor):

    pass


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0025_auto_20190930_1957'),
    ]

    operations = [
        migrations.AddField(
            model_name='hit',
            name='finished',
            field=models.NullBooleanField(),
        ),
        migrations.RunPython(forwards, backwards),
    ]
