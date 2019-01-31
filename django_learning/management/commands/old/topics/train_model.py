# import random, gensim, datetime
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from multiprocessing import cpu_count
#
# from django.core.management.base import BaseCommand, CommandError
#
# from democracy.utils import extract_sample_from_model, clean_text, decode_text, chunk_list, get_congress_stopwords, get_model_by_document_type
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
#         parser.add_argument("--stratify_by", default=None, type=str)
#         parser.add_argument("--document_type", default="press_releases", type=str)
#
#     def handle(self, *args, **options):
#
#         if options["add_defaults"]:
#             defaults = {
#                 "num_topics": 25,
#                 "sample_size": 10000,
#                 "chunk_size": 1000,
#                 "min_df": 50,
#                 "max_df": .5,
#                 "ngram_size": 1,
#                 "passes": 1,
#                 "offset": 1,
#                 "decay": .8,
#                 "num_workers": max([int(float(cpu_count())/2.0), 1]),
#                 "document_limit": None,
#                 "min_year": 2013,
#                 "no_idf": False,
#                 "stratify_by": None
#             }
#             for k in options.keys():
#                 if not options[k]:
#                     del options[k]
#             defaults.update(options)
#             options = defaults
#         del options["add_defaults"]
#         # document_limit and num_workers are run-specific, not a model property
#         document_limit = options["document_limit"]
#         num_workers = options["num_workers"]
#         chunk_size = options["chunk_size"]
#         del options["document_limit"]
#         del options["num_workers"]
#         del options["chunk_size"]
#
#         model = None
#         try:
#             model = TopicModel.objects.get(name=options["name"])
#             for opt in options:
#                 if opt in model.parameters and options[opt] != None and model.parameters[opt] != options[opt] and opt != "document_type":
#                     print "Parameter '%s' conflicts with the original value for model '%s'" % (opt, options["name"])
#                     print "Please create a new model by entering a different name, or drop this argument to use the existing value."
#                     model = None
#                     break
#         except TopicModel.DoesNotExist:
#             model = TopicModel(
#                 name=options["name"],
#                 parameters=options
#             )
#             model.save()
#
#         if model:
#
#             print model.parameters
#             if not model.vectorizer:
#
#                 print "Creating new vectorizer"
#
#                 model.vectorizer = get_vectorizer(**model.parameters)
#                 model.save()
#
#             vocab_dict = dict([(i, s) for i, s in enumerate(model.vectorizer.get_feature_names())])
#
#             if not model.model:
#                 print "Initializing new multicore LDA model"
#                 lda_model = gensim.models.ldamulticore.LdaMulticore(
#                     id2word=vocab_dict,
#                     passes=model.parameters["passes"],
#                     num_topics=model.parameters["num_topics"],
#                     chunksize=chunk_size,
#                     decay=model.parameters["decay"],
#                     offset=model.parameters["offset"],
#                     workers=num_workers
#                 )
#             else:
#                 lda_model = model.model
#
#             print "Model loaded; currently trained on %i press releases, %i tweets, and %i bills" % (model.press_releases.count(), model.tweets.count(), model.bills.count())
#             doc_model, metadata = get_model_by_document_type(options["document_type"])
#             document_ids = list(
#                 doc_model.objects\
#                     .exclude(**{"%s" % metadata["text_field"]: None})\
#                     .filter(**{"%s__gte" % metadata["date_field"]: datetime.datetime(model.parameters["min_year"], 1, 1)})\
#                     .exclude(pk__in=getattr(model, metadata["name_plural"]).values_list("pk", flat=True))\
#                     .values_list("pk", flat=True)
#             )
#
#             random.shuffle(document_ids)
#
#             if document_limit:
#                 document_ids = document_ids[:document_limit]
#             print "Model will now update using %i new %s" % (len(document_ids), metadata["name_plural"])
#
#             for i, chunk in enumerate(chunk_list(document_ids, chunk_size)):
#                 print "Updating model with chunk %i (%i total)" % (i+1, int((i+1)*chunk_size))
#                 matrix = gensim.matutils.Sparse2Corpus(
#                     model.vectorizer.transform(
#                         [
#                             clean_text(decode_text(d))
#                             for d in list(
#                                 doc_model.objects\
#                                     .filter(pk__in=chunk)\
#                                     .values_list(metadata["text_field"], flat=True)
#                             )
#                         ]
#                     ), documents_columns=False
#                 )
#                 lda_model.update(matrix)
#                 model.model = lda_model
#                 setattr(model, options["document_type"], list(set(getattr(model, options["document_type"]).values_list("pk", flat=True)).union(set(chunk))))
#                 model.save()
#
#             print "Finished updating model, saving topics"
#
#             for i in xrange(model.parameters["num_topics"]):
#                 topic = Topic.objects.create_or_update(
#                     {
#                         "name": "%s_%i" % (model.parameters["name"], i),
#                         "topic_id": i,
#                         "model": model
#                     },
#                     search_nulls=False,
#                     save_nulls=True,
#                     empty_lists_are_null=True
#                 )
#                 topic.save()
#                 topic.ngrams.all().delete()
#                 for weight, ngram in model.model.show_topic(i, topn=25):
#                     if weight >= 0.0:
#                         TopicNgram.objects.create(
#                             name=str(ngram),
#                             topic=topic,
#                             weight=weight
#                         )
#
#             for topic in Topic.objects.all():
#
#                 print str(topic)
#
#             print "Done!"
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
#     doc_model, metadata = get_model_by_document_type(options["document_type"])
#
#     if options["stratify_by"] and not hasattr(doc_model, options["stratify_by"]):
#         options["stratify_by"] = None
#         print "WARNING: the field %s does not exist on the %s model, so no stratification will occur" % (options["stratify_by"], options["document_type"])
#
#     base_index, matrix = extract_sample_from_model(
#         doc_model,
#         options["sample_size"],
#         filter_dict={"%s__gte" % metadata["date_field"]: datetime.datetime(options["min_year"], 1, 1)},
#         exclude_dict={"%s" % metadata["text_field"]: None},
#         stratify_by=options["stratify_by"],
#         output_field=metadata["text_field"],
#         text_field=True
#     )
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
# # for i, chunk in enumerate(chunk_list(press_release_ids, chunk_size)):
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
