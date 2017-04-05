from django.core.management.base import BaseCommand, CommandError

from logos.models import *


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("document_type", type=str)
        parser.add_argument("--dimensions", default=300, type=int)
        parser.add_argument("--use_skipgrams", default=False, action="store_true")
        parser.add_argument("--use_sentences", default=False, action="store_true")
        parser.add_argument("--chunk_size", default=100000, type=int)

    def handle(self, *args, **options):

        w2v = Word2VecModel.objects.create_or_update({
            "document_type": options["document_type"],
            "window_size": options["window_size"],
            "use_skipgrams": options["use_skipgrams"],
            "use_sentences": options["use_sentences"],
            "politicians_only": True
        })
        w2v.train_model(chunk_size=options["chunk_size"])