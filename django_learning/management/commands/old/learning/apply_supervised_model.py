from django.core.management.base import BaseCommand, CommandError
from django.db.models import Q

from logos.models import Document, CodeVariableClassifier


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("classifier_name", type=str)
        parser.add_argument("--num_cores", default=2, type=int)
        parser.add_argument("--chunk_size", default=1000, type=int)
        parser.add_argument("--refresh_existing", default=False, action="store_true")

    def handle(self, *args, **options):

        name = options.pop("classifier_name")
        c = CodeVariableClassifier.objects.get(name=name)

        c.apply_model_to_frames(
            num_cores=options["num_cores"],
            chunk_size=options["chunk_size"],
            refresh_existing=options["refresh_existing"]
        )