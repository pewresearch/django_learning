# -*- coding: utf-8 -*-
import os

from django.conf import settings

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

for setting, default in [
    ("DJANGO_LEARNING_HIT_TEMPLATE_DIRS", []),
    ("DJANGO_LEARNING_AWS_ACCESS", ""),
    ("DJANGO_LEARNING_AWS_SECRET", "")
]:
    if not getattr(settings, setting, None):
        globals()[setting] = default

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": globals()["DJANGO_LEARNING_HIT_TEMPLATE_DIRS"],
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