# import os, numpy
#
# from django.core.management.base import BaseCommand, CommandError
# from contextlib import closing
#
# from democracy.settings import OUTPUT_ROOT
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
#         if not os.path.exists(OUTPUT_ROOT):
#             os.mkdir(OUTPUT_ROOT)
#         export_dir = os.path.join(OUTPUT_ROOT, "exports")
#         if not os.path.exists(export_dir):
#             os.mkdir(export_dir)
#
#         with closing(open(os.path.join(OUTPUT_ROOT, "exports", "%s_topics.csv" % options["name"]), "w")) as output:
#
#             model = TopicModel.objects.get(name=options["name"])
#             output.write("model, topic_id, total, average, topic\n")
#             for i in xrange(model.parameters["num_topics"]):
#                 ngrams = []
#                 weights = []
#                 for weight, ngram in model.model.show_topic(i, topn=25):
#                     if weight >= 0.005:
#                         ngrams.append(ngram)
#                         weights.append(weight)
#                 output.write(
#                     ",".join([
#                         model.name,
#                         str(i),
#                         str(sum(weights)),
#                         str(numpy.average(weights)),
#                         " ".join(["%s*%s" % (str(n).replace(" ", "_"), str(w)[:5]) for n, w in zip(ngrams, weights)[:10] if w >= .001])
#                     ]) + "\n"
#                 )
#
#         print "Done!"