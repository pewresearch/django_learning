# -*- coding: utf-8 -*-
# Generated by Django 1.11.12 on 2019-04-08 15:41
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [("django_learning", "0020_auto_20190404_1327")]

    operations = [
        migrations.RemoveField(model_name="hittype", name="qualification_tests"),
        migrations.RemoveField(
            model_name="qualificationassignment", name="is_qualified"
        ),
    ]
