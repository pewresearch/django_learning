from django.db import models

from django_commander.models import LoggedExtendedModel
from django_learning.managers import *


class Entity(LoggedExtendedModel):
    ENTITY_TAGS = (("per", "Person"), ("org", "Organization"), ("loc", "Location"))
    name = models.CharField(max_length=300)
    tag = models.CharField(max_length=30, choices=ENTITY_TAGS)

    class Meta:
        unique_together = ("name", "tag")

    def __str__(self):
        return "{0} ({1})".format(self.name, self.tag)
