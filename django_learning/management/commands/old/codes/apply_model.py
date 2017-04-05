# import datetime, sys
#
# from django.core.management.base import BaseCommand, CommandError
# from multiprocessing.pool import Pool
#
# from democracy.models import CodeVariable, CodeVariableClassifier
# from democracy.utils import get_model_by_document_type, chunker
#
#
# class Command(BaseCommand):
#     """
#     """
#     help = ""
#
#     def add_arguments(self, parser):
#
#         parser.add_argument("--code_variable", default=None, type=str)
#         parser.add_argument("--min_year", default=2013, type=int)
#         parser.add_argument("--chunk_size", default=1000, type=int)
#         parser.add_argument("--document_type", default="press_releases", type=str)
#
#     def handle(self, *args, **options):
#
#         doc_model, metadata = get_model_by_document_type(options["document_type"])
#
#         if options["code_variable"] == "all":
#             code_vars = CodeVariable.objects.all()
#         else:
#             code_vars = [CodeVariable.objects.get(name=options["code_variable"])]
#
#         for code_variable in code_vars:
#
#             if code_variable.model:
#
#                 document_ids = list(
#                     doc_model.objects\
#                         .exclude(**{"%s" % metadata["text_field"]: None})\
#                         .filter(**{"%s" % metadata["date_field"]: datetime.datetime(options["min_year"], 1, 1)})\
#                         .values_list("pk", flat=True)
#                 )
#                 print "Model loaded; now processing %i %s" % (len(document_ids), options["document_type"])
#                 pool = Pool(processes=options["num_cores"])
#                 for i, chunk in enumerate(chunker(document_ids, options["chunk_size"])):
#                     print "Creating chunk %i of %i" % (i+1, (i+1)*options["chunk_size"])
#                     #pool.apply_async(_process_document_chunk, args=(code_variable.model.pk, chunk, i, options["document_type"]))
#                     _process_document_chunk(code_variable.model.pk, chunk, i, options["document_type"])
#                     break
#                 pool.close()
#                 pool.join()
#
#
# def _process_document_chunk(model_id, chunk, i, document_type):
#
    # try:
    #
    #     import os, django
    #     os.environ.setdefault("DJANGO_SETTINGS_MODULE", "democracy.settings")
    #     django.setup()
    #     from democracy.utils import get_model_by_document_type
    #     from django.db import connection
    #     connection.close()
    #
    #     doc_model, metadata = get_model_by_document_type(document_type)
    #     model = CodeVariableClassifier.objects.get(pk=model_id)
    #     metadata["code_model"].objects.filter(**{"%s__in" % metadata["id_field"]: chunk}).filter(code__variable__model_id=model_id).delete()
    #     docs = doc_model.objects\
    #         .filter(pk__in=chunk)\
    #         .values("pk", metadata["text_field"])
    #
    #     doc_codes = []
    #     predictions = model.model.predict([d[metadata["text_field"]] for d in docs])
    #     try: probabilities = model.model.predict_proba([d[metadata["text_field"]] for d in docs])
    #     except AttributeError: probabilities = [None] * len(docs)
    #     for doc, pred, prob in zip(docs, predictions, probabilities):
    #         doc_codes.append(
    #             metadata["code_model"](**{
    #                 "code_id": pred,
    #                 "%s" % metadata["id_field"]: doc["pk"],
    #                 "probability": prob,
    #                 "classifier": model
    #             })
    #         )
    #     metadata["code_model"].objects.bulk_create(doc_codes)
    #
    #     print "Done processing chunk %i" % (int(i)+1)
    #
    # except Exception as e:
    #
    #     print e
    #     exc_type, exc_value, exc_traceback = sys.exc_info()
    #     print exc_type
    #     print exc_value
    #     print exc_traceback