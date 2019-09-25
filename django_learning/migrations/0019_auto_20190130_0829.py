# -*- coding: utf-8 -*-
# Generated by Django 1.11.17 on 2019-01-30 08:29
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [("django_learning", "0018_auto_20180308_1057")]

    operations = [
        migrations.AlterField(
            model_name="classificationmodel",
            name="project",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="+",
                to="django_learning.Project",
            ),
        ),
        migrations.AlterField(
            model_name="coder",
            name="user",
            field=models.OneToOneField(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="coder",
                to=settings.AUTH_USER_MODEL,
            ),
        ),
        migrations.AlterField(
            model_name="question",
            name="dependency",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="dependencies",
                to="django_learning.Label",
            ),
        ),
        migrations.AlterField(
            model_name="question",
            name="project",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="questions",
                to="django_learning.Project",
            ),
        ),
        migrations.AlterField(
            model_name="question",
            name="qualification_test",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="questions",
                to="django_learning.QualificationTest",
            ),
        ),
        migrations.AlterField(
            model_name="sample",
            name="hit_type",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="samples",
                to="django_learning.HITType",
            ),
        ),
        migrations.AlterField(
            model_name="sample",
            name="parent",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="subsamples",
                to="django_learning.Sample",
            ),
        ),
    ]
