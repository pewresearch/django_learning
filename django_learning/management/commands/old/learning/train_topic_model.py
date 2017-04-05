from django.core.management.base import BaseCommand, CommandError

from logos.models import *


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("document_type", type=str)
        parser.add_argument("--num_topics", default=50, type=int)
        parser.add_argument("--sample_size", default=25000, type=int)
        parser.add_argument("--chunk_size", default=10000, type=int)

    def handle(self, *args, **options):

        tm = TopicModel.objects.create_or_update({
            "document_type": options["document_type"],
            "sample_size": options["sample_size"],
            "num_topics": options["num_topics"],
            "chunk_size": options["chunk_size"],
            "politicians_only": True
        })
        tm.save()
        tm.update_model()