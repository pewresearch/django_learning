from django.contrib.auth.models import User

from django_commander.commands import BasicCommand

from django_pewtils import get_model
from django_learning.models import Project


class Command(BasicCommand):

    parameter_names = ["coder_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("coder_name", type=str)
        return parser

    def run(self):

        # TODO:
        # so, project setup:
        # run create_coder, to create pvankessel and gstocking, with default passwords
        # in the project JSON config, just change it so that you specify the names of the admins
        # then add a tab in the project view where project admins can go through a list of expert and/or turk coders
        # and set them, using a dropdown, as either a project admin, an active coder, or inactive coder

        try:
            user = User.objects.get(username=self.parameters["coder_name"])
        except User.DoesNotExist:
            user = User.objects.create_user(
                self.parameters["coder_name"],
                "{}@pewresearch.org".format(self.parameters["coder_name"]),
                "pass"
            )
        coder = get_model("Coder").objects.create_or_update(
            {"name": self.parameters["coder_name"]},
            {"is_mturk": False, "user": user}
        )

    def cleanup(self):

        pass