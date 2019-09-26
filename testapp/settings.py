# -*- coding: utf-8 -*-
import os

SITE_NAME = "testapp"

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")).replace(
    "\\", "/"
)
APP_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__))).replace("\\", "/")
LOCAL_CACHE_ROOT = "cache"

SITE_NAME = "testapp"

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django_commander",
    "django_queries",
    "django_learning",
    "testapp",
]


DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": "postgres",
        "USER": "postgres",
        "PASSWORD": "",
        "HOST": "localhost",
        "PORT": "",
    }
}

SECRET_KEY = "testing"


##### DJANGO_LEARNING SETTINGS

DJANGO_LEARNING_BALANCING_VARIABLES = [
    os.path.join(APP_ROOT, "learning", "balancing_variables")
]
DJANGO_LEARNING_FEATURE_EXTRACTORS = [
    os.path.join(APP_ROOT, "learning", "feature_extractors")
]
DJANGO_LEARNING_PIPELINES = [os.path.join(APP_ROOT, "learning", "pipelines")]
DJANGO_LEARNING_SAMPLING_FRAMES = [
    os.path.join(APP_ROOT, "learning", "sampling_frames")
]
DJANGO_LEARNING_SAMPLING_METHODS = [
    os.path.join(APP_ROOT, "learning", "sampling_methods")
]
DJANGO_LEARNING_PROJECTS = [os.path.join(APP_ROOT, "learning", "projects")]
DJANGO_LEARNING_PROJECT_HIT_TYPES = [
    os.path.join(APP_ROOT, "learning", "project_hit_types")
]
DJANGO_LEARNING_PROJECT_QUALIFICATION_SCORERS = [
    os.path.join(APP_ROOT, "learning", "project_qualification_scorers")
]
DJANGO_LEARNING_PROJECT_QUALIFICATION_TESTS = [
    os.path.join(APP_ROOT, "learning", "project_qualification_tests")
]
DJANGO_LEARNING_HIT_TEMPLATE_DIRS = [
    os.path.join(APP_ROOT, "learning", "project_hit_templates")
]
DJANGO_LEARNING_DATASET_EXTRACTORS = [
    os.path.join(APP_ROOT, "learning", "dataset_extractors")
]
DJANGO_LEARNING_TOPIC_MODELS = [os.path.join(APP_ROOT, "learning", "topic_models")]
DJANGO_LEARNING_STOPWORD_WHITELISTS = [
    os.path.join(APP_ROOT, "learning", "stopword_whitelists")
]
DJANGO_LEARNING_REGEX_FILTERS = [os.path.join(APP_ROOT, "learning", "regex_filters")]
DJANGO_LEARNING_REGEX_REPLACERS = [
    os.path.join(APP_ROOT, "learning", "regex_replacers")
]
DJANGO_LEARNING_DATASET_CODE_FILTERS = [
    os.path.join(APP_ROOT, "learning", "dataset_code_filters")
]
DJANGO_LEARNING_EXTERNAL_PACKAGE_DIR = os.path.join(
    os.environ.get("EXTERNAL_PACKAGE_DIR", "/opt")
)
# DJANGO_LEARNING_BASE_TEMPLATE = "pyoutube/_template.html"
DJANGO_LEARNING_STOPWORD_SETS = [os.path.join(APP_ROOT, "learning", "stopword_sets")]

DJANGO_LEARNING_USE_S3 = False


### DJANGO_COMMANDER SETTINGS

DJANGO_COMMANDER_COMMAND_FOLDERS = [
    os.path.abspath(os.path.join(BASE_DIR, "testapp", "commands")).replace("\\", "/")
]
DJANGO_COMMANDER_USE_S3 = False
