import os
from django.apps import AppConfig
from django.core.exceptions import ImproperlyConfigured


DJANGO_LEARNING_BASE_DIR = os.path.dirname(os.path.realpath(__file__))


class DjangoLearningConfig(AppConfig):

    name = "django_learning"

    def update_settings(self):
        from django.conf import settings

        setattr(settings, "DJANGO_LEARNING_BASE_DIR", DJANGO_LEARNING_BASE_DIR)
        for setting, default in [
            ("AWS_ACCESS_KEY_ID", None),
            ("AWS_SECRET_ACCESS_KEY", None),
            ("DJANGO_LEARNING_BASE_TEMPLATE", "django_learning/_template.html"),
            ("S3_BUCKET", None),
            ("LOCAL_CACHE_ROOT", "cache"),
            ("S3_CACHE_ROOT", "cache"),
            ("DJANGO_LEARNING_USE_S3", False),
            ("DJANGO_LEARNING_EXTERNAL_PACKAGE_DIR", "/opt"),
        ]:
            if not hasattr(settings, setting):
                setattr(settings, setting, default)

        for setting, path in [
            ("DJANGO_LEARNING_HIT_TEMPLATE_DIRS", "templates"),
            ("DJANGO_LEARNING_BALANCING_VARIABLES", "utils/balancing_variables"),
            ("DJANGO_LEARNING_FEATURE_EXTRACTORS", "utils/feature_extractors"),
            ("DJANGO_LEARNING_PIPELINES", "utils/pipelines"),
            ("DJANGO_LEARNING_SAMPLING_FRAMES", "utils/sampling_frames"),
            ("DJANGO_LEARNING_SAMPLING_METHODS", "utils/sampling_methods"),
            ("DJANGO_LEARNING_PROJECTS", "utils/projects"),
            ("DJANGO_LEARNING_PROJECT_HIT_TYPES", "utils/project_hit_types"),
            (
                "DJANGO_LEARNING_PROJECT_QUALIFICATION_SCORERS",
                "utils/project_qualification_scorers",
            ),
            (
                "DJANGO_LEARNING_PROJECT_QUALIFICATION_TESTS",
                "utils/project_qualification_tests",
            ),
            ("DJANGO_LEARNING_DATASET_EXTRACTORS", "utils/dataset_extractors"),
            ("DJANGO_LEARNING_TOPIC_MODELS", "utils/topic_models"),
            ("DJANGO_LEARNING_STOPWORD_WHITELISTS", "utils/stopword_whitelists"),
            ("DJANGO_LEARNING_REGEX_FILTERS", "utils/regex_filters"),
            ("DJANGO_LEARNING_STOPWORD_SETS", "utils/stopword_sets"),
            ("DJANGO_COMMANDER_COMMAND_FOLDERS", "commands"),
            ("DJANGO_QUERIES_QUERY_FOLDERS", "queries/dataframes"),
            ("DJANGO_QUERIES_QUERY_FOLDERS", "queries/records"),
            ("DJANGO_QUERIES_QUERY_FOLDERS", "queries/networks"),
        ]:
            if hasattr(settings, setting):
                dirs = getattr(settings, setting)
            else:
                dirs = []
            dirs.append(os.path.join(DJANGO_LEARNING_BASE_DIR, path))
            dirs = list(set(dirs))
            setattr(settings, setting, dirs)

        LOCAL_CACHE_PATH = os.path.join(settings.LOCAL_CACHE_ROOT, "django_learning")
        setattr(settings, "DJANGO_LEARNING_LOCAL_CACHE_PATH", LOCAL_CACHE_PATH)
        S3_CACHE_PATH = os.path.join(settings.S3_CACHE_ROOT, "django_learning")
        setattr(settings, "DJANGO_LEARNING_S3_CACHE_PATH", S3_CACHE_PATH)

        templates = settings.TEMPLATES
        new_templates = []
        for template in templates:
            template["DIRS"] = (
                template["DIRS"]
                + settings.DJANGO_LEARNING_HIT_TEMPLATE_DIRS
                + [os.path.join(DJANGO_LEARNING_BASE_DIR, "templates")]
            )
            template["DIRS"] = list(set(template["DIRS"]))
            template["OPTIONS"]["context_processors"].extend(
                [
                    "django_learning.context_processors.identify_template",
                    "django_learning.context_processors.get_document_classification_model_names",
                ]
            )
            new_templates.append(template)
        setattr(settings, "TEMPLATES", new_templates)

        for dependency in ["django_queries", "django_commander"]:
            if dependency not in settings.INSTALLED_APPS:
                raise ImproperlyConfigured(
                    "{} must be in installed apps.".format(dependency)
                )

    def __init__(self, *args, **kwargs):
        super(DjangoLearningConfig, self).__init__(*args, **kwargs)
        self.update_settings()

    def ready(self):
        self.update_settings()
