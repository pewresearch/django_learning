# -*- coding: utf-8 -*-
# Generated by Django 1.10.6 on 2017-03-09 18:53
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0002_auto_20170309_1846'),
    ]

    operations = [
        migrations.AlterField(
            model_name='code',
            name='document',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='codes', to='django_learning.Document'),
        ),
        migrations.AlterField(
            model_name='code',
            name='hit',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='codes', to='django_learning.HIT'),
        ),
        migrations.AlterField(
            model_name='code',
            name='sample_unit',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, related_name='codes', to='django_learning.SampleUnit'),
        ),
    ]
