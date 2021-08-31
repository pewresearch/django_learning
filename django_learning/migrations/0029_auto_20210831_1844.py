# Generated by Django 3.1.13 on 2021-08-31 18:44

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('django_learning', '0028_auto_20201008_1633'),
    ]

    operations = [
        migrations.AlterField(
            model_name='entity',
            name='tag',
            field=models.CharField(choices=[('PERSON', 'Person, including fictional'), ('NORP', 'Nationality or religious or political groups'), ('FAC', 'Buildings, airports, highways, bridges, etc.'), ('ORG', 'Companies, agencies, institutions, etc.'), ('GPE', 'Countries, cities, states'), ('LOC', 'Non-GPE locations, mountain ranges, bodies of water'), ('PRODUCT', 'Objects, vehicles, foods, etc. (not services)'), ('EVENT', 'Named hurricanes, battles, wars, sports events, etc.'), ('WORK_OF_ART', 'Titles of books, songs, etc.'), ('LAW', 'Named documents made into laws'), ('LANGUAGE', 'Any named language'), ('DATE', 'Absolute or relative dates or periods'), ('TIME', 'Times smaller than a day'), ('PERCENT', 'Percentage'), ('MONEY', 'Monetary values, including unit'), ('QUANTITY', 'Measurements, as of weight or distance'), ('ORDINAL', 'First, second, etc'), ('CARDINAL', 'Numerals that do not fall under another type')], max_length=30),
        ),
    ]