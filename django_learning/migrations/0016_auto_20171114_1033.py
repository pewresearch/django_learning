# -*- coding: utf-8 -*-
# Generated by Django 1.11.4 on 2017-11-14 10:33
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0015_auto_20171114_1033'),
    ]

    operations = [
        migrations.AlterField(
            model_name='topicmodel',
            name='name',
            field=models.CharField(max_length=200, unique=True),
        ),
    ]
