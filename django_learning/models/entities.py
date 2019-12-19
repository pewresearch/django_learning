from django.db import models

from django_commander.models import LoggedExtendedModel
from django_learning.managers import *


class Entity(LoggedExtendedModel):
    ENTITY_TAGS = (
        ("PERSON", "Person, including fictional"),
        ("NORP", "Nationality or religious or political groups"),
        ("FAC", "Buildings, airports, highways, bridges, etc."),
        ("ORG", "Companies, agencies, institutions, etc."),
        ("GPE", "Countries, cities, states"),
        ("LOC", "Non-GPE locations, mountain ranges, bodies of water"),
        ("PRODUCT", "Objects, vehicles, foods, etc. (not services)"),
        ("EVENT", "Named hurricanes, battles, wars, sports events, etc."),
        ("WORK_OF_ART", "Titles of books, songs, etc."),
        ("LAW", "Named documents made into laws"),
        ("LANGUAGE", "Any named language"),
        ("DATE", "Absolute or relative dates or periods"),
        ("TIME", "Times smaller than a day"),
        ("PERCENT", "Percentage"),
        ("MONEY", "Monetary values, including unit"),
        ("QUANTITY", "Measurements, as of weight or distance"),
        ("ORDINAL", "First, second, etc"),
        ("CARDINAL", "Numerals that do not fall under another type"),
    )
    name = models.CharField(max_length=300)
    tag = models.CharField(max_length=30, choices=ENTITY_TAGS)

    class Meta:
        unique_together = ("name", "tag")

    def __str__(self):
        return "{0} ({1})".format(self.name, self.tag)
