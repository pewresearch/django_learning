# Generated by Django 3.1.2 on 2021-12-15 10:50

import django.contrib.postgres.fields
from django.db import migrations, models
import django.db.models.deletion
import picklefield.fields


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0033_auto_20211213_1300'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='sample',
            name='display',
        ),
        migrations.AlterField(
            model_name='documenttopic',
            name='document',
            field=models.ForeignKey(help_text='The document', on_delete=django.db.models.deletion.CASCADE, related_name='topics', to='django_learning.document'),
        ),
        migrations.AlterField(
            model_name='documenttopic',
            name='topic',
            field=models.ForeignKey(help_text='The topic', on_delete=django.db.models.deletion.CASCADE, related_name='documents', to='django_learning.topic'),
        ),
        migrations.AlterField(
            model_name='documenttopic',
            name='value',
            field=models.FloatField(help_text='Presence of the topic in the document'),
        ),
        migrations.AlterField(
            model_name='example',
            name='explanation',
            field=models.TextField(help_text='An explanation of how the text should be coded'),
        ),
        migrations.AlterField(
            model_name='example',
            name='question',
            field=models.ForeignKey(help_text='The question the example is assigned to', on_delete=django.db.models.deletion.CASCADE, related_name='examples', to='django_learning.question'),
        ),
        migrations.AlterField(
            model_name='example',
            name='quote',
            field=models.TextField(help_text='Example text'),
        ),
        migrations.AlterField(
            model_name='label',
            name='pointers',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.TextField(), default=list, help_text='List of bullet point-style tips for coders, specific to the particular label option', size=None),
        ),
        migrations.AlterField(
            model_name='label',
            name='priority',
            field=models.IntegerField(default=1, help_text='Display priority relative to other label options, lower numbers are higher priority (default is 1)'),
        ),
        migrations.AlterField(
            model_name='label',
            name='question',
            field=models.ForeignKey(help_text='The question the label belongs to', on_delete=django.db.models.deletion.CASCADE, related_name='labels', to='django_learning.question'),
        ),
        migrations.AlterField(
            model_name='label',
            name='select_as_default',
            field=models.BooleanField(default=False, help_text='(default is False) if True, this option will be selected as the default if no other option is chosen'),
        ),
        migrations.AlterField(
            model_name='qualificationassignment',
            name='coder',
            field=models.ForeignKey(help_text='The coder that took the test', on_delete=django.db.models.deletion.CASCADE, related_name='qualification_assignments', to='django_learning.coder'),
        ),
        migrations.AlterField(
            model_name='qualificationassignment',
            name='test',
            field=models.ForeignKey(help_text='The qualification test the coder took', on_delete=django.db.models.deletion.CASCADE, related_name='assignments', to='django_learning.qualificationtest'),
        ),
        migrations.AlterField(
            model_name='qualificationassignment',
            name='time_finished',
            field=models.DateTimeField(help_text='When the coder finished the test', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationassignment',
            name='time_started',
            field=models.DateTimeField(auto_now_add=True, help_text='When the coder started the test', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationassignment',
            name='turk_id',
            field=models.CharField(help_text='The assignment ID from the Mechanical Turk API (if applicable)', max_length=250, null=True),
        ),
        migrations.AlterField(
            model_name='qualificationassignment',
            name='turk_status',
            field=models.CharField(help_text='The status of the assignment in Mechanical Turk (if applicable)', max_length=40, null=True),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='approval_wait_hours',
            field=models.IntegerField(help_text='How long to wait before auto-approving Turkers (in hours)', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='coders',
            field=models.ManyToManyField(help_text='Coders that have taken the qualification test', related_name='qualification_tests', through='django_learning.QualificationAssignment', to='django_learning.Coder'),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='description',
            field=models.TextField(help_text='Description of the test (for Mechanical Turk)', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='duration_minutes',
            field=models.IntegerField(help_text='How long Turkers have to take the test (in minutes)', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='instructions',
            field=models.TextField(help_text='Instructions to be displayed at the top of the test', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='keywords',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.TextField(), default=list, help_text='List of keyword search terms (for Mechanical Turk)', size=None),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='lifetime_days',
            field=models.IntegerField(help_text='How long the test will be available (in days)', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='mturk_sandbox',
            field=models.BooleanField(default=False, help_text='(default is False) if True, the test will be created in the Mechanical Turk sandbox'),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='name',
            field=models.CharField(help_text='Unique short name for the qualification test', max_length=50),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='price',
            field=models.FloatField(help_text='How much to pay Mechanical Turk workers (in dollars)', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='title',
            field=models.TextField(help_text='Title of the test (for Mechanical Turk)', null=True),
        ),
        migrations.AlterField(
            model_name='qualificationtest',
            name='turk_id',
            field=models.CharField(help_text="Mechanical Turk ID for the test, if it's been synced via the API", max_length=250, null=True, unique=True),
        ),
        migrations.AlterField(
            model_name='question',
            name='dependency',
            field=models.ForeignKey(help_text='The label on another question that must be selected for this question to be displayed', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='dependencies', to='django_learning.label'),
        ),
        migrations.AlterField(
            model_name='question',
            name='display',
            field=models.CharField(choices=[('radio', 'radio'), ('checkbox', 'checkbox'), ('dropdown', 'dropdown'), ('text', 'text'), ('header', 'header')], help_text='The format of the question and how it will be displayed', max_length=20),
        ),
        migrations.AlterField(
            model_name='question',
            name='multiple',
            field=models.BooleanField(default=False, help_text='Whether or not multiple label selections are allowed'),
        ),
        migrations.AlterField(
            model_name='question',
            name='name',
            field=models.CharField(help_text='Short name of the question, must be unique to the project or qualification test', max_length=250),
        ),
        migrations.AlterField(
            model_name='question',
            name='optional',
            field=models.BooleanField(default=False, help_text='(default is False) if True, coders will be able to skip the question'),
        ),
        migrations.AlterField(
            model_name='question',
            name='priority',
            field=models.IntegerField(default=1, help_text='Order in which the question should be displayed relative to other questions. This gets set automatically based on the JSON config but can be modified manually. Lower numbers are higher priority.'),
        ),
        migrations.AlterField(
            model_name='question',
            name='project',
            field=models.ForeignKey(help_text='The project the question belongs to', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='questions', to='django_learning.project'),
        ),
        migrations.AlterField(
            model_name='question',
            name='prompt',
            field=models.TextField(help_text='The prompt that will be displayed to coders'),
        ),
        migrations.AlterField(
            model_name='question',
            name='qualification_test',
            field=models.ForeignKey(help_text='The qualification test the question belongs to', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='questions', to='django_learning.qualificationtest'),
        ),
        migrations.AlterField(
            model_name='question',
            name='show_notes',
            field=models.BooleanField(default=False, help_text='(default is False) if True, coders can write and submit notes about their decisions regarding this specific question'),
        ),
        migrations.AlterField(
            model_name='question',
            name='tooltip',
            field=models.TextField(help_text='Optional text to be displayed when coders hover over the question', null=True),
        ),
        migrations.AlterField(
            model_name='sample',
            name='documents',
            field=models.ManyToManyField(help_text='Documents in the sample', related_name='samples', through='django_learning.SampleUnit', to='django_learning.Document'),
        ),
        migrations.AlterField(
            model_name='sample',
            name='frame',
            field=models.ForeignKey(help_text='The sampling frame the sample was drawn from', on_delete=django.db.models.deletion.CASCADE, related_name='samples', to='django_learning.samplingframe'),
        ),
        migrations.AlterField(
            model_name='sample',
            name='name',
            field=models.CharField(help_text='A name for the sample (must be unique within the project)', max_length=100),
        ),
        migrations.AlterField(
            model_name='sample',
            name='parent',
            field=models.ForeignKey(help_text="The parent sample, if it's a subsample", null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='subsamples', to='django_learning.sample'),
        ),
        migrations.AlterField(
            model_name='sample',
            name='project',
            field=models.ForeignKey(help_text='The coding project the sample belongs to', on_delete=django.db.models.deletion.CASCADE, related_name='samples', to='django_learning.project'),
        ),
        migrations.AlterField(
            model_name='sample',
            name='qualification_tests',
            field=models.ManyToManyField(help_text='Qualification tests required to code the sample (set automatically)', related_name='samples', to='django_learning.QualificationTest'),
        ),
        migrations.AlterField(
            model_name='sample',
            name='sampling_method',
            field=models.CharField(default='random', help_text='The method used for sampling (must correspond to a sampling method file)', max_length=200, null=True),
        ),
        migrations.AlterField(
            model_name='sampleunit',
            name='document',
            field=models.ForeignKey(help_text='The document that was sampled', on_delete=django.db.models.deletion.CASCADE, related_name='sample_units', to='django_learning.document'),
        ),
        migrations.AlterField(
            model_name='sampleunit',
            name='sample',
            field=models.ForeignKey(help_text='The sample the document belongs to', on_delete=django.db.models.deletion.CASCADE, related_name='document_units', to='django_learning.sample'),
        ),
        migrations.AlterField(
            model_name='sampleunit',
            name='weight',
            field=models.FloatField(default=1.0, help_text='Sampling weight based on the sample and sampling frame the document belongs to'),
        ),
        migrations.AlterField(
            model_name='samplingframe',
            name='documents',
            field=models.ManyToManyField(help_text='The documents that belong to the sampling frame', related_name='sampling_frames', to='django_learning.Document'),
        ),
        migrations.AlterField(
            model_name='samplingframe',
            name='name',
            field=models.CharField(help_text='Name of the sampling frame (must correspond to a config file)', max_length=200, unique=True),
        ),
        migrations.AlterField(
            model_name='topic',
            name='anchors',
            field=django.contrib.postgres.fields.ArrayField(base_field=models.CharField(max_length=100), default=list, help_text='Optional list of terms to be used as anchors for this topic when training the model', size=None),
        ),
        migrations.AlterField(
            model_name='topic',
            name='label',
            field=models.CharField(db_index=True, help_text='Optional label/title given to the topic', max_length=300, null=True),
        ),
        migrations.AlterField(
            model_name='topic',
            name='model',
            field=models.ForeignKey(help_text='The model the topic belongs to (unique together with ``num``)', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='topics', to='django_learning.topicmodel'),
        ),
        migrations.AlterField(
            model_name='topic',
            name='name',
            field=models.CharField(db_index=True, help_text='Optional short name given to the topic', max_length=50, null=True),
        ),
        migrations.AlterField(
            model_name='topic',
            name='num',
            field=models.IntegerField(help_text='Unique number of the topic in the model (unique together with ``model``)'),
        ),
        migrations.AlterField(
            model_name='topicmodel',
            name='frame',
            field=models.ForeignKey(help_text='Sampling frame the topic model belongs to', on_delete=django.db.models.deletion.CASCADE, related_name='topic_models', to='django_learning.samplingframe'),
        ),
        migrations.AlterField(
            model_name='topicmodel',
            name='model',
            field=picklefield.fields.PickledObjectField(editable=False, help_text='Pickled topic model', null=True),
        ),
        migrations.AlterField(
            model_name='topicmodel',
            name='name',
            field=models.CharField(help_text='Unique name for the topic model', max_length=200, unique=True),
        ),
        migrations.AlterField(
            model_name='topicmodel',
            name='training_documents',
            field=models.ManyToManyField(help_text='Documents the model was trained on', related_name='topic_models_trained', to='django_learning.Document'),
        ),
        migrations.AlterField(
            model_name='topicmodel',
            name='vectorizer',
            field=picklefield.fields.PickledObjectField(editable=False, help_text='Pickled vectorizer', null=True),
        ),
        migrations.AlterField(
            model_name='topicngram',
            name='name',
            field=models.CharField(db_index=True, help_text='The ngram', max_length=40),
        ),
        migrations.AlterField(
            model_name='topicngram',
            name='topic',
            field=models.ForeignKey(help_text='The topic the ngram belongs to', on_delete=django.db.models.deletion.CASCADE, related_name='ngrams', to='django_learning.topic'),
        ),
        migrations.AlterField(
            model_name='topicngram',
            name='weight',
            field=models.FloatField(help_text="Weight indicating 'how much' the ngram belongs to the topic"),
        ),
    ]