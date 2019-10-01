# -*- coding: utf-8 -*-
# Generated by Django 1.11.24 on 2019-09-26 12:07
from __future__ import unicode_literals

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [("django_learning", "0024_auto_20190822_1549")]

    operations = [
        migrations.CreateModel(
            name="MovieReview",
            fields=[
                (
                    "id",
                    models.AutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                (
                    "document",
                    models.OneToOneField(
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="movie_review",
                        to="django_learning.Document",
                    ),
                ),
            ],
            options={"abstract": False},
        )
    ]