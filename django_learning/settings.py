# -*- coding: utf-8 -*-
import os

from django.conf import settings
from django.db import models

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
# Copied from django_extensions as a boilerplate example
# REPLACEMENTS = {
# }
# add_replacements = getattr(settings, 'EXTENSIONS_REPLACEMENTS', {})
# REPLACEMENTS.update(add_replacements)

from django.db import models


if not getattr(settings, 'DJANGO_LEARNING_BASE_MODEL', None):
    from django_commander.models import LoggedExtendedModel
    DJANGO_LEARNING_BASE_MODEL = LoggedExtendedModel
    # from pewtils.django.abstract_models import BasicExtendedModel
    # DJANGO_LEARNING_BASE_MODEL = BasicExtendedModel
    # DJANGO_LEARNING_BASE_MODEL = models.Model
if not getattr(settings, 'DJANGO_LEARNING_BASE_MANAGER', None):
    from pewtils.django.managers import BasicManager
    DJANGO_LEARNING_BASE_MANAGER = BasicManager
    #DJANGO_LEARNING_BASE_MANAGER = models.QuerySet
if not getattr(settings, "DJANGO_LEARNING_HIT_TEMPLATE_DIRS", None):
    DJANGO_LEARNING_HIT_TEMPLATE_DIRS = []

# APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)).decode('utf-8')).replace('\\', '/')
# TEMPLATE_ROOT = os.path.abspath(os.path.join(APP_ROOT, "templates").decode('utf-8')).replace('\\', '/')

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": DJANGO_LEARNING_HIT_TEMPLATE_DIRS,
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages'
            ]
        }
    }
]


INSTALLED_APPS = ('django_commander', )
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)).decode('utf-8')).replace('\\', '/')
DJANGO_COMMANDER_COMMAND_FOLDERS = [
    os.path.abspath(os.path.join(APP_ROOT, "commands").decode('utf-8')).replace('\\', '/')
]
