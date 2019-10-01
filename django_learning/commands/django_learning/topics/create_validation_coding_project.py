from __future__ import print_function, absolute_import
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
        "sampling_searches": {
            '"""

sampling_method_template_middle = """': {
                "pattern": \""""

sampling_method_template_end = """\",
                "proportion": .5
            }
        }
    }
"""

regex_filter_template_start = """
import re

def get_regex():

    return re.compile(r\""""

regex_filter_template_end = """\", re.IGNORECASE)"""


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
            sampling_method_template_middle,
            "topic_model_{}".format(topic_model.name),
            sampling_method_template_end,
        ]
    )
    with closing(
        open(
            os.path.join(
                settings.DJANGO_LEARNING_SAMPLING_METHODS[0],
                "topic_model_{}.py".format(topic_model.name),
            ),
            "wb",
        )
    ) as output:
        output.write(sampling_method_text)

    regex_filter_text = "".join(
        [regex_filter_template_start, regex, regex_filter_template_end]
    )
    with closing(
        open(
            os.path.join(
                settings.DJANGO_LEARNING_REGEX_FILTERS[0],
                "topic_model_{}.py".format(topic_model.name),
            ),
            "wb",
        )
    ) as output:
        output.write(regex_filter_text)


class Command(BasicCommand):

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
            with closing(
                open(
                    os.path.join(
                        settings.DJANGO_LEARNING_PROJECTS[0],
                        "topic_model_{}.json".format(topic_model.name),
                    ),
                    "wb",
                )
            ) as output:
                json.dump(project, output, indent=4)

            if self.options["reset_project"]:
                Project.objects.get(
                    name="topic_model_{}".format(topic_model.name)
                ).delete()

            commands["create_project"](
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

                commands["extract_sample"](
                    project_name=project.name,
                    hit_type_name=hit_type.name,
                    sample_name=project.name,
                    sampling_frame_name=topic_model.frame.name,
                    sampling_method="topic_model_{}".format(topic_model.name),
                    size=self.options["sample_size"],
                ).run()
                commands["create_sample_hits_experts"](
                    project_name=project.name,
                    sample_name=project.name,
                    num_coders=self.options["num_coders"],
                ).run()

            else:

                print(
                    "Sample already extracted for '{}'".format(
                        "topic_model_{}".format(topic_model.name)
                    )
                )
