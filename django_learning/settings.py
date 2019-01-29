# -*- coding: utf-8 -*-
import os

from django.conf import settings

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

for setting, default in [
    ("DJANGO_LEARNING_HIT_TEMPLATE_DIRS", []),
    ("AWS_ACCESS_KEY_ID", None),
    ("AWS_SECRET_ACCESS_KEY", None),
    ("DJANGO_LEARNING_BASE_TEMPLATE", "django_learning/_template.html"),
    ("S3_BUCKET", ""),
    ("LOCAL_CACHE_ROOT", ""),
    ("S3_CACHE_ROOT", "")
]:
    value = getattr(settings, setting, None)

    if value is None and default is not None:
        globals()[setting] = default

    elif value is not None:
        globals()[setting] = value

LOCAL_CACHE_PATH = os.path.join(globals()["LOCAL_CACHE_ROOT"], "django_learning")
globals()["LOCAL_CACHE_PATH"] = LOCAL_CACHE_PATH

S3_CACHE_PATH = os.path.join(globals()["S3_CACHE_ROOT"], "django_learning")
globals()["S3_CACHE_PATH"] = S3_CACHE_PATH

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": globals()["DJANGO_LEARNING_HIT_TEMPLATE_DIRS"] + [os.path.join(BASE_DIR, 'templates')],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django_learning.context_processors.identify_template',
                'django_learning.context_processors.get_document_classification_model_names'
            ]
        }
    }
]

INSTALLED_APPS = ('django_commander', )
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__))).replace('\\', '/')
DJANGO_COMMANDER_COMMAND_FOLDERS = [
    os.path.abspath(os.path.join(APP_ROOT, "commands")).replace('\\', '/')
]

#### DJANGO_QUERIES SETTINGS

DJANGO_QUERIES_QUERY_FOLDERS = [
    os.path.abspath(os.path.join(APP_ROOT, "queries", "dataframes")).replace('\\', '/'),
    os.path.abspath(os.path.join(APP_ROOT, "queries", "records")).replace('\\', '/'),
    os.path.abspath(os.path.join(APP_ROOT, "queries", "networks")).replace('\\', '/')
]
