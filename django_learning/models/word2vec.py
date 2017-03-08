import numpy

from django.db import models

from picklefield.fields import PickledObjectField

from django_learning.utils.preprocessors.clean_text import Preprocessor as CleanTextPreprocessor
from django_learning.settings import DJANGO_LEARNING_BASE_MODEL, DJANGO_LEARNING_BASE_MANAGER
from django_learning.utils import get_document_types

from pewtils import chunker
from pewtils.django import get_model
from pewtils.nlp import TextCleaner, SentenceTokenizer


class Word2VecModel(DJANGO_LEARNING_BASE_MODEL):
    # document_type = models.CharField(max_length=60, choices=get_document_types(), help_text="The type of document")
    window_size = models.IntegerField(default=10)
    use_skipgrams = models.BooleanField(default=False)
    use_sentences = models.BooleanField(default=False)
    dimensions = models.IntegerField(default=300)
    frame = models.ForeignKey("django_learning.SamplingFrame", related_name="word2vec_models", null=True)

    finalized = models.BooleanField(default=False)

    model = PickledObjectField(null=True)

    training_documents = models.ManyToManyField("django_learning.Document", related_name="word2vec_models_trained")

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    # class Meta:
    #
    #   unique_together = ("document_type", "window_size", "use_skipgrams", "use_sentences", "dimensions")

    def __str__(self):

        return "{}, window_size={}, use_sentences={}, use_skipgrams={}, dimensions={}, finalized={}".format(
            self.document_type,
            self.window_size,
            self.use_sentences,
            self.use_skipgrams,
            self.dimensions,
            self.finalized
        )

    def train_model(self, chunk_size=100000, workers=2, doc_limit=None):

        if not self.finalized:

            cleaner = TextCleaner(lemmatize=False)
            tokenizer = SentenceTokenizer()
            w2v_model = self.model

            print "Extracting document IDs"
            if self.frame:
                doc_ids = self.frame.documents \
                    .filter(is_clean=True) \
                    .filter(text__isnull=False)
            else:
                doc_ids = get_model("Document").objects \
                    .filter(**{"{}__isnull".format(self.document_type): False}) \
                    .filter(is_clean=True) \
                    .filter(text__isnull=False)
            doc_ids = list(doc_ids.values_list("pk", flat=True))
            random.shuffle(doc_ids)

            for i, chunk in tqdm(enumerate(chunker(doc_ids, chunk_size)), desc="Processing document chunks"):

                chunk_docs = get_model("Document").objects.filter(pk__in=chunk)

                sentences = []
                for text in tqdm(chunk_docs.values_list("text", flat=True),
                                 nested=True,
                                 desc="Loading documents {0} - {1}".format((i) * chunk_size, (i + 1) * chunk_size)):
                    if self.use_sentences:
                        sentences.extend([cleaner.clean(s).split() for s in tokenizer.tokenize(text)])
                    else:
                        sentences.append(cleaner.clean(text).split())
                print "Transforming and training"
                bigram_transformer = gensim.models.Phrases(sentences)

                if not w2v_model:
                    w2v_model = Word2Vec(
                        bigram_transformer[sentences],
                        size=self.dimensions,
                        sg=1 if self.use_skipgrams else 0,
                        window=self.window_size,
                        min_count=5,
                        workers=workers
                    )
                else:
                    w2v_model.train(bigram_transformer[sentences])

                print "{0} documents loaded, saving model".format((i + 1) * chunk_size)

                self.model = w2v_model
                self.save()
                self.training_documents.add(*chunk_docs)

                if doc_limit and (i + 1) * chunk_size >= doc_limit:
                    break

            print "Finished updating model"

    def finalize_model(self):

        w2v_model = self.model
        w2v_model.init_sims(replace=True)
        self.model = w2v_model
        self.finalized = True
        self.save()

        print "Finalized model"

    def clear_model(self):

        self.model = None
        self.save()

    def apply_model_to_docs(self, docs, stopword_sets=None, regex_filters=None):

        df = pandas.DataFrame.from_records(docs.values("pk", "text"))
        return self.apply_model_to_dataframe(df, stopword_sets=stopword_sets, regex_filters=regex_filters)

    def apply_model_to_dataframe(self, df, stopword_sets=None, regex_filters=None):

        if not stopword_sets: stopword_sets = []
        if not regex_filters: regex_filters = []

        cleaner = CleanTextPreprocessor(lemmatize=False, stopword_sets=stopword_sets, regex_filters=regex_filters)
        df['text'] = df['text'].map(cleaner.run)

        model = self.model

        vecs = []
        for index, row in tqdm(df.iterrows(), desc="Computing Word2Vec features"):
            text = row['text']
            new_row = []
            words = []
            for word in text.split():
                try:
                    wordvec = model[word]
                except KeyError:
                    wordvec = None
                if wordvec != None:
                    words.append(wordvec)
            if len(words) == 0:
                new_row.extend([numpy.nan] * self.dimensions * 6)
            else:
                words = pandas.DataFrame(words)
                new_row.extend(
                    list(words.mean()) +
                    list(words.max()) +
                    list(words.min()) +
                    list(words.apply(lambda x: numpy.median(x))) +
                    list(words.apply(lambda x: numpy.percentile(x, 25))) +
                    list(words.apply(lambda x: numpy.percentile(x, 75)))
                )
            vecs.append(new_row)

        return pandas.DataFrame(vecs, columns=self.get_feature_names())

    def get_feature_names(self):

        return ["{}_avg".format(i) for i in range(0, self.dimensions)] + \
               ["{}_max".format(i) for i in range(0, self.dimensions)] + \
               ["{}_min".format(i) for i in range(0, self.dimensions)] + \
               ["{}_median".format(i) for i in range(0, self.dimensions)] + \
               ["{}_25pct".format(i) for i in range(0, self.dimensions)] + \
               ["{}_75pct".format(i) for i in range(0, self.dimensions)]


        # class Word2VecModelDocument(LoggedExtendedModel):

#
#     word2vec_model = models.ForeignKey("django_learning.Word2VecModel", related_name="documents")
#     document = models.ForeignKey("django_learning.Document", related_name="word2vec_models")
#     features = ArrayField(models.FloatField(), default=[])
# TODO: store W2V features in arrayfields; write "apply_model' function for w2v models (and same for ngramsets and topics)
# TODO: modify feature extractors to use precomputed DB values instead of local caching (and to compute and save them when needed)
# TODO: what about preprocessors though?  should those be saved as immutable params on the DB objects?
# the model shouldn't be trained on preprocessed data, except maybe... lowercasing?
