# -*- coding: utf-8 -*-
# Generated by Django 1.10.6 on 2017-03-10 10:35
from __future__ import unicode_literals

import django.contrib.postgres.fields
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0003_auto_20170309_1853'),
    ]

    operations = [
        migrations.AddField(
            model_name='label',
            name='pointers',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.TextField(), default=[], size=None),
        ),
    ]
