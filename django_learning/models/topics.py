import numpy, pandas, gensim, random, os

from django.db import models

from picklefield.fields import PickledObjectField

from django_commander.models import LoggedExtendedModel
from django_learning.utils import get_document_types

from pewtils.django import get_model
from pewtils.django.sampling import SampleExtractor


class TopicModel(LoggedExtendedModel):

    frame = models.ForeignKey("django_learning.SamplingFrame", related_name="topic_models")
    num_topics = models.IntegerField(default=100)
    decay = models.FloatField(default=.8)
    offset = models.IntegerField(default=1)
    passes = models.IntegerField(default=1)
    sample_size = models.IntegerField(default=10000)
    chunk_size = models.IntegerField(default=1000)

    workers = models.IntegerField(default=2)

    model = PickledObjectField(null=True)
    vectorizer = PickledObjectField(null=True)

    training_documents = models.ManyToManyField("django_learning.Document", related_name="topic_models_trained")

    class Meta:

      unique_together = ("frame", "num_topics", "decay", "offset", "passes", "sample_size", "chunk_size")

    def __str__(self):

        return "{}, num_topics={}, decay={}, offset={}, passes={}, sample_size={}, chunk_size={}".format(
            self.frame,
            self.decay,
            self.offset,
            self.passes,
            self.sample_size,
            self.chunk_size
        )

    def _get_document_ids(self):

        doc_ids = list(
            self.frame.documents \
                .filter(is_clean=True) \
                .filter(text__isnull=False) \
                .values_list("pk", flat=True)
        )

        return doc_ids

    def save(self, *args, **kwargs):

        if not self.vectorizer:
            print "Initializing new topic model ({}, {})".format(self.frame, self.num_topics)

            from django_learning.utils.feature_extractors import feature_extractors

            self.vectorizer = feature_extractors['tfidf'](
                sublinear_tf=False,
                max_df=.6,
                min_df=max([10, int(float(self.sample_size) * .0005)]),
                max_features=10000,
                ngram_range=(1, 2),
                use_idf=True,
                norm="l2",
                preprocessors=[
                    ("clean_text", {
                        "lemmatize": True,
                        "regex_filters": [],
                        "stopword_sets": ["english", "months", "misc_boilerplate"]
                    })
                ]
            )

            print "Extracting sample for vectorizer"

            frame_ids = self._get_document_ids()
            random.shuffle(frame_ids)

            frame_ids = pandas.DataFrame({"pk": frame_ids})
            sample_ids = SampleExtractor(id_col="pk", sampling_strategy="random").extract(frame_ids, self.sample_size)
            sample = pandas.DataFrame.from_records(
                get_model("Document").objects.filter(pk__in=sample_ids).values("pk", "text"))

            print "Training vectorizer on {} documents".format(len(sample))
            self.vectorizer = self.vectorizer.fit(sample)
            print "{} features extracted from vectorizer".format(len(self.vectorizer.get_feature_names()))

        super(TopicModel, self).save(*args, **kwargs)

    def update_model(self, doc_limit=None, recycle_existing=False):

        vocab_dict = dict([(i, s) for i, s in enumerate(self.vectorizer.get_feature_names())])

        lda_model = self.model
        if not lda_model:
            lda_model = gensim.models.ldamulticore.LdaMulticore(
                chunksize=self.chunk_size,
                passes=self.passes,
                decay=self.decay,
                offset=self.offset,
                num_topics=self.num_topics,
                workers=self.workers,
                id2word=vocab_dict
            )

        print "Extracting document IDs"
        doc_ids = self._get_document_ids()
        if not recycle_existing:
            doc_ids = list(set(doc_ids) - set(self.training_documents.values_list("pk", flat=True)))
        random.shuffle(doc_ids)

        for i, chunk in tqdm(enumerate(chunker(doc_ids, self.chunk_size)), desc="Processing document chunks"):

            print "Updating model with chunk %i (%i total)" % (i + 1, int((i + 1) * self.chunk_size))
            chunk_docs = get_model("Document").objects.filter(pk__in=chunk)
            matrix = self.vectorizer.transform(
                pandas.DataFrame.from_records(chunk_docs.values("pk", "text"))
            )
            matrix = gensim.matutils.Sparse2Corpus(matrix, documents_columns=False)
            lda_model.update(matrix)

            self.model = lda_model
            self.save()
            self.training_documents.add(*chunk_docs)

            print self.model.show_topics(self.num_topics)

            self.update_topics()

            if doc_limit and (i + 1) * self.chunk_size >= doc_limit:
                break

        print "Finished updating model"

    def update_topics(self):

        for i in xrange(self.num_topics):
            topic = Topic.objects.create_or_update(
                {
                    "num": i,
                    "model": self
                },
                search_nulls=False,
                save_nulls=True,
                empty_lists_are_null=True
            )
            topic.ngrams.all().delete()
            for weight, ngram in self.model.show_topic(i, topn=25):
                if weight >= 0.0:
                    TopicNgram.objects.create(
                        name=str(ngram),
                        topic=topic,
                        weight=weight
                    )

        print "Topics updated"

    def apply_model(self, min_probability=.5, doc_limit=None):

        doc_ids = self._get_document_ids()
        random.shuffle(doc_ids)
        if doc_limit: doc_ids = doc_ids[:doc_limit]

        lda_model = self.model
        for i, chunk in tqdm(enumerate(chunker(doc_ids, self.chunk_size)), desc="Processing document chunks"):
            print "Updating model with chunk %i (%i total)" % (i + 1, int((i + 1) * self.chunk_size))
            chunk_docs = get_model("Document").objects.filter(pk__in=chunk)
            matrix = self.vectorizer.transform(
                pandas.DataFrame.from_records(chunk_docs.values("pk", "text"))
            )
            matrix = gensim.matutils.Sparse2Corpus(matrix, documents_columns=False)
            for doc, bow in zip(chunk_docs, matrix):
                doc.topics.filter(topic__model=self).delete()
                doc_topics = lda_model.get_document_topics(bow, minimum_probability=min_probability)
                for topic, weight in doc_topics:
                    get_model("DocumentTopic").objects.create_or_update(
                        {"document": doc, "topic": self.topics.get(num=topic), "value": weight},
                        return_object=False,
                        save_nulls=True,
                        search_nulls=False
                    )

                    # def top_n_documents(self, n=10, document_type=None):
                    #     doc_topics = DocumentTopic.objects.all()
                    #     if document_type: doc_topics = doc_topics.filter(**{"document__{0}_id__isnull".format(document_type): False})
                    #     return doc_topics.filter(topic__model=self).order_by("-value")[:n]
                    #
                    # def topic_distribution(self):
                    #
                    #     df = pandas.DataFrame(
                    #         list(DocumentTopic.objects.filter(topic__model=self).values("topic_id", "value"))
                    #     )
                    #     data = []
                    #     for index, row in df.groupby("topic_id").agg({"value": numpy.average}).sort("value").iterrows():
                    #         data.append((Topic.objects.get(pk=index), row["value"]))
                    #     return data


class Topic(LoggedExtendedModel):
    """
    A topic extracted by a topic model.
    """

    num = models.IntegerField()
    model = models.ForeignKey("django_learning.TopicModel", related_name="topics", null=True, on_delete=models.SET_NULL)
    label = models.CharField(max_length=100, db_index=True, null=True)

    class Meta:

        unique_together = ("num", "model")

    def __str__(self):

        if self.label:
            return self.label
        else:
            return self.top_ngrams()

    def set_label(self, label):

        self.label = label
        self.save()

    def top_ngrams(self):

        return " ".join([n.replace(" ", "_") for n in
                         self.ngrams.order_by("-weight").filter(weight__gte=.001).values_list("name", flat=True)[:10]])

    def distinctive_ngrams(self):

        return " ".join([ngram.name.replace(" ", "_") for ngram in self.ngrams.order_by("-weight") if
                         get_model("TopicNgram").objects.filter(name=ngram.name).filter(
                             topic__model=self.model).order_by("-weight")[0].topic == self])

    def most_similar(self, with_value=False):
        sim = self.similar_topics()
        if len(sim) > 0:
            if with_value:
                return sim[0]
            else:
                return sim[0][0]
        else:
            return None

    def values(self):
        return list(self.documents.values_list("value", flat=True))

    def values_at_least_1pct(self):
        return list(self.documents.filter(value__gte=.01).values_list("value", flat=True))

    def top_n_documents(self, n=10):
        return self.documents.order_by("-value")[:n]

    def top_n_tweets(self, n=10):
        return self.tweets.order_by("-value")[:n]

    def top_n_bills(self, n=10):
        return self.bills.order_by("-value")[:n]

    def coef_avg(self):
        return numpy.average(list(self.ngrams.values_list("weight", flat=True)))

    def coef_total(self):
        return sum(list(self.ngrams.values_list("weight", flat=True)))

    def coef_std(self):
        return numpy.std(list(self.ngrams.values_list("weight", flat=True)))

    def document_avg(self):
        return numpy.average(list(self.documents.values_list("value", flat=True)))

    def document_total(self):
        return sum(list(self.documents.values_list("value", flat=True)))

    def document_std(self):
        return numpy.std(list(self.documents.values_list("value", flat=True)))


class TopicNgram(LoggedExtendedModel):
    name = models.CharField(max_length=40, db_index=True)
    topic = models.ForeignKey("django_learning.Topic", related_name="ngrams")
    weight = models.FloatField()

    def __str__(self):
        return "{}*{}".format(self.name, self.weight)


class DocumentTopic(LoggedExtendedModel):
    topic = models.ForeignKey("django_learning.Topic", related_name="documents")
    document = models.ForeignKey("django_learning.Document", related_name="topics")
    value = models.FloatField()

    class Meta:
        unique_together = ("topic", "document")

    def __str__(self):
        return "{0}, {1}: {2}".format(
            str(self.document),
            str(self.topic),
            self.value
        )
