# import random, datetime, pandas
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from multiprocessing import cpu_count
# from multiprocessing.pool import Pool
#
# from django.core.management.base import BaseCommand, CommandError
#
# from democracy.utils import clean_text, decode_text, chunker, get_congress_stopwords, get_model_by_document_type
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
#         if not options["num_workers"]:
#             options["num_workers"] = max([int(float(cpu_count())/2.0), 1])
#         model = TopicModel.objects.get(name=options["name"])
#         topics = {int(t.topic_id): str(t.pk) for t in model.topics.exclude(label__isnull=True)}
#
#         doc_model, metadata = get_model_by_document_type(options["document_type"])
#         document_ids = list(
#             doc_model.objects\
#                 .exclude(**{"%s" % metadata["text_field"]: None})\
#                 .filter(**{"%s__gte" % metadata["date_field"]: datetime.datetime(model.parameters["min_year"], 1, 1)})\
#                 # .exclude(pk__in=getattr(model, metadata["name_plural"]).values_list("pk", flat=True))\
#                 .values_list("pk", flat=True)
#         )
#
#         print "Model loaded; now processing %i %s" % (len(document_ids), options["document_type"])
#         pool = Pool(processes=options["num_workers"])
#         for i, chunk in enumerate(chunker(document_ids, options["chunk_size"])):
#             print "Creating chunk %i of %i" % (i+1, (i+1)*options["chunk_size"])
#             pool.apply_async(_process_document_chunk, args=(model.pk, chunk, topics, i, options["document_type"]))
#             # _process_document_chunk(model.pk, chunk, topics, i, options["document_type"])
#             # break
#         pool.close()
#         pool.join()
#
#
# def _process_document_chunk(model_id, chunk, topics, i, document_type):
#
#     try:
#
#         import gensim, os, django, sys
#         os.environ.setdefault("DJANGO_SETTINGS_MODULE", "democracy.settings")
#         django.setup()
#         from democracy.utils import clean_text, decode_text
#         from democracy.models import PressRelease, PressReleaseTopic, TopicModel, Tweet, Bill, TweetTopic, BillTopic
#         from democracy.utils import get_document_type_info
#         from django.db import connection
#         connection.close()
#
#         #doc_model, doctopic_model, doccode_model, id_name, doc_text, doc_date = get_document_type_info(document_type)
#         doc_model, metadata = get_model_by_document_type(document_type)
#
#         model = TopicModel.objects.get(pk=model_id)
#         matrix = gensim.matutils.Sparse2Corpus(
#             model.vectorizer.transform(
#                 [
#                     clean_text(decode_text(d))
#                     for d in list(
#                         doc_model.objects\
#                             .filter(pk__in=chunk)\
#                             .values_list(metadata["text_field"], flat=True)
#                     )
#                 ]
#             ), documents_columns=False
#         )
#
#         topic_set = set(topics.keys())
#
#         metadata["topic_model"].objects.filter(**{"%s__in" % metadata["id_field"]: chunk}).filter(topic__model_id=model_id).delete()
#         doc_topics = []
#         for doc_id, row in zip(chunk, gensim.matutils.corpus2csc(model.model[matrix]).transpose()):
#             doc_topics.extend([
#                 metadata["topic_model"](**{
#                     "topic_id": topics[topic_id],
#                     "%s" % metadata["id_field"]: doc_id,
#                     "value": float(value)
#                 }) for topic_id, value in zip(row.indices, row.data) if topic_id in topics
#             ])
#             doc_topics.extend([
#                 metadata["topic_model"](**{
#                     "topic_id": topics[topic_id],
#                     "%s" % metadata["id_field"]: doc_id,
#                     "value": 0.0
#                 }) for topic_id in topic_set.difference(set(row.indices)) if topic_id in topics
#             ])
#         metadata["topic_model"].objects.bulk_create(doc_topics)
#
#         print "Done processing chunk %i" % (int(i)+1)
#
#     except Exception as e:
#
#         print e
#         exc_type, exc_value, exc_traceback = sys.exc_info()
#         print exc_type
#         print exc_value
#         print exc_traceback
#
#
# def get_vectorizer(**options):
#
#     print "Extracting vectorizer training sample"
#
#     vectorizer = TfidfVectorizer(
#         sublinear_tf=False,
#         max_df=options["max_df"],
#         min_df=options["min_df"],
#         ngram_range=(1, options["ngram_size"]),
#         use_idf=(not options["no_idf"]),
#         stop_words=get_congress_stopwords(add_english=True)
#     )
#
#     base_index, matrix = [], []
#
#     doc_model, metadata = get_model_by_document_type(options["document_type"])
#
#     if hasattr(doc_model, "politician"):
#
#         docs = pandas.DataFrame(
#             list(
#                 doc_model.objects\
#                     .exclude(**{"%s" % metadata["text_field"]: None})\
#                     .filter(**{"%s__gte" % metadata["date_field"]: datetime.datetime(options["min_year"], 1, 1)})\
#                     .values("pk", "politician_id")
#             )
#         )
#         proportion = float(options["sample_size"]) / float(len(docs['politician_id'].unique()))
#         ids = []
#         for r in docs['politician_id'].unique():
#             p_ids = list(docs[docs['politician_id']==r]['pk'])
#             random.shuffle(p_ids)
#             p_ids = p_ids[:int(proportion)]
#             ids.extend(p_ids)
#             print "Extracted %i %s for pol %s" % (len(p_ids), options["document_type"], r)
#         print "Stratified sample of %i %s extracted; cleaning text" % (len(ids), options["document_type"])
#         for doc in doc_model.objects.filter(pk__in=ids):
#             base_index.append(doc.pk)
#             matrix.append(clean_text(decode_text(getattr(doc, metadata["text_field"]))))
#
#     else:
#
#         print "Extracting %s" % options["document_type"]
#         doc_ids = doc_model.objects\
#             .exclude(**{"%s" % metadata["text_field"]: None})\
#             .filter(**{"%s__gte" % metadata["date_field"]: datetime.datetime(options["min_year"], 1, 1)})\
#             .values_list("pk", flat=True)
#         print "Sample extracted; cleaning text"
#         random.shuffle([i for i in doc_ids])
#         doc_ids = doc_ids[:options["sample_size"]]
#         for doc in doc_model.objects.filter(pk__in=doc_ids):
#             base_index.append(doc.pk)
#             matrix.append(clean_text(decode_text(getattr(doc, metadata["text_field"]))))
#
#     print "Training vectorizer"
#     vectorizer = vectorizer.fit(matrix)
#
#     return vectorizer
#
#
# # ---- Handy references:
# # http://radimrehurek.com/gensim/wiki.html
# # https://github.com/piskvorky/gensim/blob/develop/gensim/models/ldamodel.py
# # https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore
# # https://radimrehurek.com/gensim/tut1.html
#
#
# # ---- Old code, etc.
#
#
# # ---- Install help for VW
# # sudo git clone https://github.com/JohnLangford/vowpal_wabbit.git
# # cd vowpal_wabbit
# # sudo git checkout tags/7.10
# # sudo yum install libtool
# # sudo touch README
# # sudo autoreconf --force --install
# # sudo ./configure --with-boost-libdir=/usr/lib/x86_64-linux-gnu
# # sudo make
#
#
# # class StreamingPressReleaseCorpus(object):
# #
# #     def __init__(self, vectorizer=None, min_doc_length=100):
# #
# #         print "Initializing StreamingPressReleaseCorpus"
# #         self.ids = []
# #         if not vectorizer:
# #             vectorizer = get_vectorizer()
# #         self.vectorizer = vectorizer
# #         self.vocab = self.vectorizer.get_feature_names()
# #         self.vocab_dict = dict([(i, s) for i, s in enumerate(self.vocab)])
# #         self.min_doc_length=min_doc_length
# #         self.press_release_ids = PressRelease.objects.exclude(content=None).values_list("pk", flat=True)[:1000].iterator()
# #
# #     def __iter__(self):
# #
# #         return self
# #
# #     def next(self):
# #
# #         print "Stream started"
# #         # valid_press_release = False
# #         # while not valid_press_release:
# #         for press_release_id in self.press_release_ids:
# #             press_release = PressRelease.objects.get(pk=press_release_id)
# #             # press_release = self.press_releases.next()
# #             text = clean_text(decode_text(press_release["content"]))
# #             if len(text) >= self.min_doc_length:
# #                 vectors = self.vectorizer.transform([text])
# #                 if vectors[0].sum() > 0.0:
# #                     corpus = gensim.matutils.Sparse2Corpus(vectors, documents_columns=False)
# #                     corpus_list = list(corpus)
# #                     if len(corpus_list) > 0 and len(corpus_list[0]) > 0:
# #                         yield corpus
# #                         self.ids.append(press_release["pk"])
# #                         if len(self.ids) % 50 == 0:
# #                             print "%i press releases processed" % len(self.ids)
#
#
# # for i, topic in enumerate(model.show_topics(num_topics=num_topics, num_words=25, formatted=False)):
#
# # model = gensim.models.ldamodel.LdaModel(
# #     "/vowpal_wabbit/vowpalwabbit/vw",
# #     num_topics=topic_cnt,
# #     id2word=vocab,
# #     tmp_prefix="vw_lda_tmp",
# #     cleanup_files=True,
# #     chunksize=chunk_size
# # )
# #
# # press_release_ids = PressRelease.objects.exclude(clean_text=None).values_list("pk", flat=True)
# # for i, chunk in enumerate(chunker(press_release_ids, chunk_size)):
# #     print "Updating model with chunk %i (%i total)" % (i+1, (i+1)*chunk_size)
# #     matrix = gensim.matutils.Sparse2Corpus(vectorizer.transform(list(PressRelease.objects.filter(pk__in=chunk).values_list("clean_text", flat=True))), documents_columns=False)
# #     model.update(matrix)
# #     print "Saving topic model and vectorizer"
# #     model.save("vw_lda.model")
#
# # for t in model.show_topics(num_topics=topic_cnt, num_words=5):
# #     if float(t[:5]) > 0:
# #         print t
#
#
# # press_releases["text"] = press_releases["content"].map(lambda x: "| %s\n" % clean_text(decode_text(x)))
# #
# # print "Creating input file"
# # ids = []
# # with closing(open(os.path.join(export_dir, "lda_input.vw"), "wb")) as input_file:
# #     for index, row in press_releases.iterrows():
# #         try:
# #             input_file.write(row["text"])
# #             ids.append(row["pk"])
# #         except Exception as e:
# #             print e
# #
# # print "Computing LDA model"
# # try:
# #     output = subprocess.check_output([
# #         "vw",
# #         "-d", os.path.join(export_dir, "lda_input.vw"),
# #         "--lda", str(num_topics),
# #         "--lda_D", "9999999",
# #         "--lda_alpha", str(lda_alpha),
# #         "--passes", str(passes),
# #         "--minibatch", str(batch_size),
# #         "--readable_model", os.path.join(export_dir, "lda_model.vw")
# #     ], stderr=subprocess.STDOUT)
# #     errors = None
# # except Exception as e:
# #     exc_type, exc_value, exc_traceback = sys.exc_info()
# #     if exc_type == subprocess.CalledProcessError:
# #         errors = e.output
# #     else:
# #         errors = "%s, %s, %s" % (str(exc_type), str(exc_value), str(traceback.format_exc()))
# #     print errors
#
#
# # vw = VW(
# #     moniker='lda',
# #     passes=passes,
# #     lda=num_topics,
# #     minibatch=batch_size,
# #     lda_alpha=lda_alpha,
# #     decay_learning_rate=decay_learning_rate,
# #     working_dir=os.path.join(export_dir)
# # )
# # with vw.training():
# #     for index, row in press_releases.iterrows():
# #         vw.push_instance("| %s\n" % row["text"])
# #         ids.append(row["pk"])
#
