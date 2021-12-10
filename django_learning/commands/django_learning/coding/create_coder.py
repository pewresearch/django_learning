from django.contrib.auth.models import User

from django_commander.commands import BasicCommand

from django_pewtils import get_model
from django_learning.models import Project


class Command(BasicCommand):

    """
    Create a coder in the database. Defaults to password 'pass' (Django Learning is assuming that it's deployed in
    a safe environment, and user accounts are purely for tracking coding data.)

    :param coder_name: Username for the coder to be created
    """

    parameter_names = ["coder_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("coder_name", type=str)
        return parser

    def run(self):

        try:
            user = User.objects.get(username=self.parameters["coder_name"])
        except User.DoesNotExist:
            user = User.objects.create_user(
                self.parameters["coder_name"],
                "{}@pewresearch.org".format(self.parameters["coder_name"]),
                "pass",
            )
        get_model("Coder").objects.create_or_update(
            {"name": self.parameters["coder_name"]}, {"is_mturk": False, "user": user}
        )

    def cleanup(self):

        pass
