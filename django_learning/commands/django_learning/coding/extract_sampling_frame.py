from __future__ import print_function
from django_commander.commands import BasicCommand

from django_learning.models import SamplingFrame


class Command(BasicCommand):

    parameter_names = ["sampling_frame_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("sampling_frame_name", type=str)
        parser.add_argument("--refresh", default=False, action="store_true")
        return parser

    def run(self):

        frame, created = SamplingFrame.objects.get_or_create(name=self.parameters['sampling_frame_name'])
        frame.extract_documents(refresh=self.options['refresh'])
        if self.options["refresh"]:
            frame.get_sampling_flags(refresh=True)
        print(frame)

    def cleanup(self):

        pass