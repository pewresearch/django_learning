# -*- coding: utf-8 -*-
# Generated by Django 1.11.4 on 2017-11-13 17:01
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0013_auto_20171113_1038'),
    ]

    operations = [
        migrations.AlterField(
            model_name='topic',
            name='label',
            field=models.CharField(db_index=True, max_length=300, null=True),
        ),
    ]
