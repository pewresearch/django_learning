from django_commander.commands import BasicCommand

from django_learning.models import SamplingFrame


class Command(BasicCommand):

    option_defaults = [
        {"name": "refresh", "default": False, "type": bool}
    ]
    parameter_defaults = [
        {"name": "sampling_frame_name", "default": None, "type": str}
    ]
    dependencies = []

    def run(self):

        frame, created = SamplingFrame.objects.get_or_create(name=self.parameters['sampling_frame_name'])
        frame.extract_documents(refresh=self.options['refresh'])
        print frame

    def cleanup(self):

        pass