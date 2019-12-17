from django_commander.commands import BasicCommand

from django_learning.models import SamplingFrame, TopicModel


class Command(BasicCommand):

    parameter_names = ["name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("name", type=str)
        parser.add_argument("--refresh_model", action="store_true", default=False)
        parser.add_argument("--refresh_vectorizer", action="store_true", default=False)
        return parser

    def run(self):

        model = TopicModel.objects.create_or_update({"name": self.parameters["name"]})
        model.save()
        model.load_model(
            refresh_model=self.options["refresh_model"],
            refresh_vectorizer=self.options["refresh_vectorizer"],
        )

    def cleanup(self):

        pass
