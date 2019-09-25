# -*- coding: utf-8 -*-
# Generated by Django 1.11.4 on 2017-11-13 10:38
from __future__ import unicode_literals

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [("django_learning", "0011_auto_20171109_1357")]

    operations = [
        migrations.AlterUniqueTogether(name="documenttopic", unique_together=set([])),
        migrations.RemoveField(model_name="documenttopic", name="command_logs"),
        migrations.RemoveField(model_name="documenttopic", name="commands"),
        migrations.RemoveField(model_name="documenttopic", name="document"),
        migrations.RemoveField(model_name="documenttopic", name="topic"),
        migrations.AlterUniqueTogether(name="topic", unique_together=set([])),
        migrations.RemoveField(model_name="topic", name="command_logs"),
        migrations.RemoveField(model_name="topic", name="commands"),
        migrations.RemoveField(model_name="topic", name="model"),
        migrations.AlterUniqueTogether(name="topicmodel", unique_together=set([])),
        migrations.RemoveField(model_name="topicmodel", name="command_logs"),
        migrations.RemoveField(model_name="topicmodel", name="commands"),
        migrations.RemoveField(model_name="topicmodel", name="frame"),
        migrations.RemoveField(model_name="topicmodel", name="training_documents"),
        migrations.RemoveField(model_name="topicngram", name="command_logs"),
        migrations.RemoveField(model_name="topicngram", name="commands"),
        migrations.RemoveField(model_name="topicngram", name="topic"),
        migrations.DeleteModel(name="DocumentTopic"),
        migrations.DeleteModel(name="Topic"),
        migrations.DeleteModel(name="TopicModel"),
        migrations.DeleteModel(name="TopicNgram"),
    ]
