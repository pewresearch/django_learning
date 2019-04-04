# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..").decode('utf-8')).replace('\\', '/')
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)).decode('utf-8')).replace('\\', '/')

SITE_NAME = "testapp"

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django_commander",
    "django_learning",
    "testapp"
]


DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'postgres',
        'USER': 'postgres',
        'PASSWORD': None,
        'HOST': 'localhost',
        'PORT': '',
    }
}

SECRET_KEY = "testing"

##### DJANGO_COMMANDER SETTINGS

DJANGO_COMMANDER_COMMAND_FOLDERS = [
    os.path.abspath(os.path.join(APP_ROOT, "commands").decode('utf-8')).replace('\\', '/')
]


##### DJANGO_LEARNING SETTINGS

DJANGO_LEARNING_BALANCING_VARIABLES = [os.path.join(APP_ROOT, "learning", "balancing_variables")]
DJANGO_LEARNING_FEATURE_EXTRACTORS = [os.path.join(APP_ROOT, "learning", "feature_extractors")]
DJANGO_LEARNING_PIPELINES = [os.path.join(APP_ROOT, "learning", "pipelines")]
DJANGO_LEARNING_SAMPLING_FRAMES = [os.path.join(APP_ROOT, "learning", "sampling", "frames")]
DJANGO_LEARNING_SAMPLING_METHODS = [os.path.join(APP_ROOT, "learning", "sampling", "methods")]
DJANGO_LEARNING_PROJECTS = [os.path.join(APP_ROOT, "learning", "projects")]
DJANGO_LEARNING_PROJECT_HIT_TYPES = [os.path.join(APP_ROOT, "learning", "project_hit_types")]
DJANGO_LEARNING_PROJECT_QUALIFICATION_SCORERS = [os.path.join(APP_ROOT, "learning", "project_qualification_scorers")]
DJANGO_LEARNING_PROJECT_QUALIFICATION_TESTS = [os.path.join(APP_ROOT, "learning", "project_qualification_tests")]
DJANGO_LEARNING_HIT_TEMPLATE_DIRS = [os.path.join(APP_ROOT, "learning", "project_hit_templates")]
DJANGO_LEARNING_DATASET_EXTRACTORS = [os.path.join(APP_ROOT, "learning", "dataset_extractors")]
DJANGO_LEARNING_TOPIC_MODELS = [os.path.join(APP_ROOT, "learning", "topic_models")]
DJANGO_LEARNING_STOPWORD_WHITELISTS = [os.path.join(APP_ROOT, "learning", "stopword_whitelists")]
DJANGO_LEARNING_REGEX_FILTERS = [os.path.join(APP_ROOT, "learning", "regex_filters")]
DJANGO_LEARNING_EXTERNAL_PACKAGE_DIR = os.path.join(os.environ.get("EXTERNAL_PACKAGE_DIR", "/opt"))
# DJANGO_LEARNING_BASE_TEMPLATE = "pyoutube/_template.html"
DJANGO_LEARNING_STOPWORD_SETS = [os.path.join(APP_ROOT, "learning", "stopword_sets")]

DJANGO_LEARNING_AWS_ACCESS = os.environ.get("AWS_ACCESS_KEY_ID", None)
DJANGO_LEARNING_AWS_SECRET = os.environ.get("AWS_SECRET_ACCESS_KEY", None)


##### URL AND TEMPLATE SETTINGS

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': DJANGO_LEARNING_HIT_TEMPLATE_DIRS, # TODO: you shouldn't have to add this, django_learning should add it, right?
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'django_learning.context_processors.identify_template',
                'django_learning.context_processors.get_document_classification_model_names'
            ]
        }
    },
]