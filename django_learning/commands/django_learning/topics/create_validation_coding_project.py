from __future__ import print_function
import os
import gensim, json

from contextlib import closing

from django_commander.commands import BasicCommand, log_command, commands
from django_learning.models import *


sampling_method_template_start = """
import pandas

from django.db.models import F
from pewtils import is_null
from django_pewtils import get_model


def get_method():
    return {
        "sampling_strategy": "random",
        "stratify_by": None, 
        "sampling_searches": [{"regex_filter": \""""

sampling_method_template_end = """\", "proportion": .5}]
    }
"""

regex_filter_template_start = """
import re

def get_regex():

    return re.compile(r\""""

regex_filter_template_end = """\", re.IGNORECASE)"""


def _save_to_file(topic_model, folder_name, text, file_ext):

    import django_learning

    path = None
    for f in getattr(settings, folder_name):
        if os.path.dirname(django_learning.__file__) not in f:
            path = f
            break
    if is_null(path):
        raise Exception(
            "Your app needs to specify a folder in settings.{}".format(folder_name)
        )
    outpath = os.path.join(path, "topic_model_{}.{}".format(topic_model.name, file_ext))
    print("Saving to {}".format(outpath))
    with closing(open(outpath, "w")) as output:
        if file_ext == "json":
            json.dump(text, output, indent=4)
        else:
            output.write(text)


def create_topic_sampling_method(topic_model):

    anchors = []
    for topic in topic_model.topics.filter(label__isnull=False).exclude(label=""):
        anchors.extend(topic.anchors)
    anchors = list(set(anchors))
    regex = r"|".join([r"\b{}\b".format(ngram) for ngram in anchors])

    sampling_method_text = "".join(
        [
            sampling_method_template_start,
            "topic_model_{}".format(topic_model.name),
            sampling_method_template_end,
        ]
    )
    _save_to_file(
        topic_model, "DJANGO_LEARNING_SAMPLING_METHODS", sampling_method_text, "py"
    )

    regex_filter_text = "".join(
        [regex_filter_template_start, regex, regex_filter_template_end]
    )
    _save_to_file(topic_model, "DJANGO_LEARNING_REGEX_FILTERS", regex_filter_text, "py")


class Command(BasicCommand):

    """
    Command to set up a coding project to validate a topic model once it's been trained and finalized. You can first
    run this command with ``create_project_files=True`` to create and save a codebook file automatically. The codebook
    will consist of a question for each labeled topic in the model, asking whether or not the document mentions that
    topic or not. The command will also make a custom sampling method that pulls a sample where 50% of the documents
    match to at least one of the anchor terms in the model.

    Once you've created the project file, you can then run the command without the ``--create_project_files`` option,
    and specify a ``sample_size`` and ``num_coders`` and coding samples and HITs will be created automatically for you,
    oversampled on the topic model's anchor terms.

    :param topic_model_name: Name of an existing topic model
    :param admin_name: Name of the Coder who will be set up as the project admin
    :param project_hit_type: Name of the HIT type to use on the project
    :param sample_size: (default is 100) size of the validation sample to pull
    :param num_coders: (default is 2) number of coders to complete each HIT
    :param reset_project: (default is False) if True, deletes the project and recreates it if it already exists
    :param create_project_files: (default is False) if True, compiles a codebook file automatically and saves it to \
        the ``settings.DJANGO_LEARNING_PROJECTS[0]`` folder with the name ``topic_model_[TOPIC_MODEL_NAME].json``
    """

    parameter_names = ["topic_model_name", "admin_name", "project_hit_type"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("topic_model_name", type=str)
        parser.add_argument("admin_name", type=str)
        parser.add_argument("project_hit_type", type=str)
        parser.add_argument("--sample_size", type=int, default=100)
        parser.add_argument("--num_coders", type=int, default=2)
        parser.add_argument("--reset_project", action="store_true", default=False)
        parser.add_argument(
            "--create_project_files", action="store_true", default=False
        )
        return parser

    @log_command
    def run(self):

        topic_model = get_model("TopicModel", app_name="django_learning").objects.get(
            name=self.parameters["topic_model_name"]
        )

        print(topic_model.name)

        if self.options["create_project_files"]:

            project = {
                "instructions": "",
                "qualification_tests": [],
                "admins": [self.parameters["admin_name"]],
                "coders": [{"name": self.parameters["admin_name"], "is_admin": True}],
                "questions": [
                    {
                        "prompt": "Does this response mention any of the following themes?",
                        "name": "topic_header",
                        "display": "header",
                    }
                ],
            }
            for topic in topic_model.topics.filter(label__isnull=False).exclude(
                label=""
            ):
                project["questions"].append(
                    {
                        "prompt": topic.label,
                        "name": topic.name,
                        "display": "checkbox",
                        "labels": [
                            {
                                "label": "No",
                                "value": "0",
                                "pointers": [],
                                "select_as_default": True,
                            },
                            {"label": "Yes", "value": "1", "pointers": []},
                        ],
                        "tooltip": "",
                        "examples": [],
                    }
                )
            _save_to_file(topic_model, "DJANGO_LEARNING_PROJECTS", project, "json")

            if self.options["reset_project"]:
                Project.objects.get(
                    name="topic_model_{}".format(topic_model.name)
                ).delete()

            commands["django_learning_coding_create_project"](
                project_name="topic_model_{}".format(topic_model.name)
            ).run()

            project = Project.objects.get(
                name="topic_model_{}".format(topic_model.name)
            )
            hit_type = HITType.objects.create_or_update(
                {"name": self.parameters["project_hit_type"], "project": project}
            )

            create_topic_sampling_method(topic_model)

        else:

            project = Project.objects.get(
                name="topic_model_{}".format(topic_model.name)
            )
            hit_type = HITType.objects.create_or_update(
                {"name": self.parameters["project_hit_type"], "project": project}
            )

            if (
                Sample.objects.filter(
                    name="topic_model_{}".format(topic_model.name), project=project
                ).count()
                == 0
            ):

                from django_learning.utils import regex_filters, sampling_methods

                reload(regex_filters)
                reload(sampling_methods)

                commands["django_learning_coding_extract_sample"](
                    project_name=project.name,
                    sample_name=project.name,
                    sampling_frame_name=topic_model.frame.name,
                    sampling_method="topic_model_{}".format(topic_model.name),
                    size=self.options["sample_size"],
                ).run()
                commands["django_learning_coding_create_sample_hits"](
                    project_name=project.name,
                    sample_name=project.name,
                    hit_type_name=hit_type.name,
                    num_coders=self.options["num_coders"],
                ).run()

            else:

                print(
                    "Sample already extracted for '{}'".format(
                        "topic_model_{}".format(topic_model.name)
                    )
                )
