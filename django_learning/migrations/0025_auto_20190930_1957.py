# Generated by Django 2.2.4 on 2019-09-30 19:57

import django.contrib.postgres.fields
import django.contrib.postgres.fields.jsonb
from django.db import migrations, models
import picklefield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0024_auto_20190822_1549'),
    ]

    operations = [
        migrations.AlterField(
            model_name='classification',
            name='probability',
            field=models.FloatField(help_text='The probability of the assigned label, if applicable', null=True),
        ),
        migrations.AlterField(
            model_name='classificationmodel',
            name='name',
            field=models.CharField(help_text='Unique name of the classifier', max_length=100, unique=True),
        ),
        migrations.AlterField(
            model_name='classificationmodel',
            name='parameters',
            field=picklefield.fields.PickledObjectField(editable=False, help_text='A pickle file of the parameters used to process the codes and generate the model', null=True),
        ),
        migrations.AlterField(
            model_name='classificationmodel',
            name='pipeline_name',
            field=models.CharField(help_text="The named pipeline used to seed the handler's parameters, if any; note that the JSON pipeline file may have changed since this classifier was created; refer to the parameters field to view the exact parameters used to compute the model", max_length=150, null=True),
        ),
        migrations.AlterField(
            model_name='code',
            name='date_added',
            field=models.DateTimeField(auto_now_add=True, help_text='The date the document code was added'),
        ),
        migrations.AlterField(
            model_name='code',
            name='date_last_updated',
            field=models.DateTimeField(auto_now=True, help_text='The last date the document code was modified'),
        ),
        migrations.AlterField(
            model_name='coder',
            name='is_mturk',
            field=models.BooleanField(default=False, help_text='Whether or not the coder is a Mechanical Turk worker'),
        ),
        migrations.AlterField(
            model_name='coder',
            name='name',
            field=models.CharField(help_text='Unique name of the coder', max_length=200, unique=True),
        ),
        migrations.AlterField(
            model_name='document',
            name='alternative_text',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.TextField(), default=list, size=None),
        ),
        migrations.AlterField(
            model_name='document',
            name='date',
            field=models.DateTimeField(help_text='An optional date associated with the document', null=True),
        ),
        migrations.AlterField(
            model_name='document',
            name='duplicate_ids',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.IntegerField(), default=list, size=None),
        ),
        migrations.AlterField(
            model_name='document',
            name='text',
            field=models.TextField(help_text='The text content of the document'),
        ),
        migrations.AlterField(
            model_name='documentfragment',
            name='scope',
            field=django.contrib.postgres.fields.jsonb.JSONField(default=dict, help_text='A dictionary of filter parameters for defining documents within which the fragment can exist'),
        ),
        migrations.AlterField(
            model_name='entity',
            name='tag',
            field=models.CharField(choices=[('per', 'Person'), ('org', 'Organization'), ('loc', 'Location')], max_length=30),
        ),
        migrations.AlterField(
            model_name='hittype',
            name='keywords',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.TextField(), default=list, size=None),
        ),
        migrations.AlterField(
            model_name='label',
            name='label',
            field=models.CharField(help_text='A longer label for the code value', max_length=400),
        ),
        migrations.AlterField(
            model_name='label',
            name='pointers',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.TextField(), default=list, size=None),
        ),
        migrations.AlterField(
            model_name='label',
            name='value',
            field=models.CharField(db_index=True, help_text='The code value', max_length=50),
        ),
        migrations.AlterField(
            model_name='ngramset',
            name='words',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=50), default=list, size=None),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='keywords',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.TextField(), default=list, size=None),
        ),
        migrations.AlterField(
            model_name='question',
            name='display',
            field=models.CharField(choices=[('radio', 'radio'), ('checkbox', 'checkbox'), ('dropdown', 'dropdown'), ('text', 'text'), ('header', 'header')], max_length=20),
        ),
        migrations.AlterField(
            model_name='sample',
            name='display',
            field=models.CharField(choices=[('article', 'Article'), ('image', 'Image'), ('audio', 'Audio'), ('video', 'Video')], max_length=20),
        ),
        migrations.AlterField(
            model_name='sample',
            name='sampling_method',
            field=models.CharField(default='random', max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='topic',
            name='anchors',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=100), default=list, size=None),
        ),
        migrations.AlterField(
            model_name='topicmodel',
            name='parameters',
            field=picklefield.fields.PickledObjectField(editable=False, help_text='A pickle file of the parameters used', null=True),
        ),
    ]
