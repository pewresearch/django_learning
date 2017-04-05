# from django.core.management.base import BaseCommand, CommandError
#
# from democracy.models import *
#
#
# class Command(BaseCommand):
#
#     help = ""
#
#     def add_arguments(self, parser):
#
#         parser.add_argument("name")
#         parser.add_argument("--num_topics", default=None, type=int)
#         parser.add_argument("--sample_size", default=None, type=int)
#         parser.add_argument("--chunk_size", default=None, type=int)
#         parser.add_argument("--min_df", default=None, type=int)
#         parser.add_argument("--max_df", default=None, type=float)
#         parser.add_argument("--ngram_size", default=None, type=int)
#         parser.add_argument("--passes", default=None, type=int)
#         parser.add_argument("--offset", default=None, type=int)
#         parser.add_argument("--decay", default=None, type=float)
#         parser.add_argument("--num_workers", default=None, type=int)
#         parser.add_argument("--document_limit", default=None, type=int)
#         parser.add_argument("--min_year", default=None, type=int)
#         parser.add_argument("--no_idf", default=None, action="store_false")
#         parser.add_argument("--add_defaults", default=False, action="store_true")
#         parser.add_argument("--document_type", default="press_releases", type=str)
#
#     def handle(self, *args, **options):
#
#         m = TopicModel.objects.get(name=options["name"])
#         for t in m.topics.all():
#             print t.top_ngrams()
#             print "Current label: %s" % (t.label if t.label else "---")
#             new_label = raw_input("Enter a label (or click enter to keep the same) >>> ")
#             if new_label.strip() not in ["", None]:
#                 t.label = new_label
#                 t.save()