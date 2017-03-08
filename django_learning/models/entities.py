from django.db import models

from django_learning.settings import DJANGO_LEARNING_BASE_MODEL, DJANGO_LEARNING_BASE_MANAGER
from django_learning.managers import *


class Entity(DJANGO_LEARNING_BASE_MODEL):
    ENTITY_TAGS = (
        ("per", "Person"),
        ("org", "Organization"),
        ("loc", "Location")
    )
    name = models.CharField(max_length=300)
    tag = models.CharField(max_length=30, choices=ENTITY_TAGS)

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    class Meta:
        unique_together = ("name", "tag")

    def __str__(self):
        return "{0} ({1})".format(self.name, self.tag)

