from __future__ import print_function
import numpy, pandas, gensim, random, os

from django.db import models
from django.contrib.postgres.fields import ArrayField

from picklefield.fields import PickledObjectField
from corextopic import corextopic as corex_topic

from django_commander.models import LoggedExtendedModel
from django_learning.utils import get_document_types
from django_learning.utils import topic_models
from django_learning.models import SamplingFrame

from pewtils import is_null, is_not_null
from django_pewtils import get_model
from django_pewtils.sampling import SampleExtractor


class TopicModel(LoggedExtendedModel):

    name = models.CharField(max_length=200, unique=True)
    frame = models.ForeignKey("django_learning.SamplingFrame", related_name="topic_models", on_delete=models.CASCADE)

    model = PickledObjectField(null=True)
    vectorizer = PickledObjectField(null=True)

    parameters = PickledObjectField(null=True, help_text="A pickle file of the parameters used")

    training_documents = models.ManyToManyField("django_learning.Document", related_name="topic_models_trained")

    def __str__(self):

        return self.name

    def save(self, *args, **kwargs):

        self.parameters = topic_models.topic_models[self.name]()
        self.frame = SamplingFrame.objects.get(name=self.parameters["frame"])
        super(TopicModel, self).save(*args, **kwargs)

    def _get_document_ids(self):

        doc_ids = list(
            self.frame.documents \
                .filter(text__isnull=False) \
                .values_list("pk", flat=True)
        )

        return doc_ids

    def load_model(self, refresh_model=False, refresh_vectorizer=False):

        # vocab_dict = dict([(i, s) for i, s in enumerate(self.vectorizer.get_feature_names())])
        if refresh_model or is_null(self.model):

            if refresh_vectorizer or is_null(self.vectorizer):

                from django_learning.utils.feature_extractors import feature_extractors

                self.vectorizer = feature_extractors['tfidf'](**self.parameters["vectorizer"])

                print("Extracting sample for vectorizer")

                frame_ids = self._get_document_ids()
                random.shuffle(frame_ids)

                if not self.parameters["sample_size"]:
                    sample_ids = frame_ids
                else:
                    sample_ids = SampleExtractor(id_col="pk", sampling_strategy="random").extract(
                        pandas.DataFrame({"pk": frame_ids}), self.parameters["sample_size"]
                    )
                self.training_documents = get_model("Document").objects.filter(pk__in=sample_ids)
                sample = pandas.DataFrame.from_records(self.training_documents.values("pk", "text"))

                print("Training vectorizer on {} documents".format(len(sample)))
                self.vectorizer = self.vectorizer.fit(sample)
                print("{} features extracted from vectorizer".format(len(self.vectorizer.get_feature_names())))

            print("Initializing new topic model ({}, {})".format(self.frame, self.parameters["num_topics"]))
            self.model = corex_topic.Corex(n_hidden=self.parameters["num_topics"], seed=42)
            tfidf = self.vectorizer.transform(
                pandas.DataFrame.from_records(self.training_documents.values("pk", "text"))
            )

            ngrams = self.vectorizer.get_feature_names()

            old_topics = list(self.topics.values())

            anchors = []
            anchor_topic_num = 0
            old_topic_map = {}
            for t in old_topics:
                if is_not_null(t['anchors']):
                    anchor_list = [anchor for anchor in t['anchors'] if anchor in ngrams]
                    if len(anchor_list) > 0:
                        anchors.append(anchor_list)
                        old_topic_map[anchor_topic_num] = t
                        anchor_topic_num += 1

            self.model = self.model.fit(
                tfidf,
                words=self.vectorizer.get_feature_names(),
                anchors=anchors,
                anchor_strength=self.parameters["anchor_strength"]
            )

            self.save()

            self.topics.all().delete()
            for i, topic_ngrams in enumerate(self.model.get_topics(n_words=20)):

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
                for ngram, weight in topic_ngrams:
                    TopicNgram.objects.create(
                        name=str(ngram),
                        topic=topic,
                        weight=weight
                    )
                print(str(topic))

                old_topic = old_topic_map.get(i, None)
                if old_topic:

                    topic.name = old_topic['name']
                    topic.label = old_topic['label']
                    topic.anchors = old_topic['anchors']
                    topic.save()

    def apply_model(self, df, probabilities=False):

        tfidf = self.vectorizer.transform(df)
        topic_names = list(self.topics.order_by("num").values_list("name", flat=True))
        if not probabilities: topic_df = pandas.DataFrame(self.model.transform(tfidf), columns=topic_names).astype(int)
        else: topic_df = pandas.DataFrame(self.model.transform(tfidf, details=True)[1], columns=topic_names).astype(float)
        topic_df.index = df.index
        return pandas.concat([df, topic_df], axis=1)


        # lda_model = self.model
        # if not lda_model:
        #     lda_model = gensim.models.ldamulticore.LdaMulticore(
        #         chunksize=self.chunk_size,
        #         passes=self.passes,
        #         decay=self.decay,
        #         offset=self.offset,
        #         num_topics=self.num_topics,
        #         workers=self.workers,
        #         id2word=vocab_dict
        #     )
        #
        # print "Extracting document IDs"
        # doc_ids = self._get_document_ids()
        # if not recycle_existing:
        #     doc_ids = list(set(doc_ids) - set(self.training_documents.values_list("pk", flat=True)))
        # random.shuffle(doc_ids)
        #
        # for i, chunk in tqdm(enumerate(chunk_list(doc_ids, self.chunk_size)), desc="Processing document chunks"):
        #
        #     print "Updating model with chunk %i (%i total)" % (i + 1, int((i + 1) * self.chunk_size))
        #     chunk_docs = get_model("Document").objects.filter(pk__in=chunk)
        #     matrix = self.vectorizer.transform(
        #         pandas.DataFrame.from_records(chunk_docs.values("pk", "text"))
        #     )
        #     matrix = gensim.matutils.Sparse2Corpus(matrix, documents_columns=False)
        #     lda_model.update(matrix)
        #
        #     self.model = lda_model
        #     self.save()
        #     self.training_documents.add(*chunk_docs)
        #
        #     print self.model.show_topics(self.num_topics)
        #
        #     self.update_topics()
        #
        #     if doc_limit and (i + 1) * self.chunk_size >= doc_limit:
        #         break
        #
        # print "Finished updating model"

    # def apply_model(self, min_probability=.5, doc_limit=None):
    #
    #     doc_ids = self._get_document_ids()
    #     random.shuffle(doc_ids)
    #     if doc_limit: doc_ids = doc_ids[:doc_limit]
    #
    #     lda_model = self.model
    #     for i, chunk in tqdm(enumerate(chunk_list(doc_ids, self.chunk_size)), desc="Processing document chunks"):
    #         print "Updating model with chunk %i (%i total)" % (i + 1, int((i + 1) * self.chunk_size))
    #         chunk_docs = get_model("Document").objects.filter(pk__in=chunk)
    #         matrix = self.vectorizer.transform(
    #             pandas.DataFrame.from_records(chunk_docs.values("pk", "text"))
    #         )
    #         matrix = gensim.matutils.Sparse2Corpus(matrix, documents_columns=False)
    #         for doc, bow in zip(chunk_docs, matrix):
    #             doc.topics.filter(topic__model=self).delete()
    #             doc_topics = lda_model.get_document_topics(bow, minimum_probability=min_probability)
    #             for topic, weight in doc_topics:
    #                 get_model("DocumentTopic").objects.create_or_update(
    #                     {"document": doc, "topic": self.topics.get(num=topic), "value": weight},
    #                     return_object=False,
    #                     save_nulls=True,
    #                     search_nulls=False
    #                 )
    #
    #                 # def top_n_documents(self, n=10, document_type=None):
    #                 #     doc_topics = DocumentTopic.objects.all()
    #                 #     if document_type: doc_topics = doc_topics.filter(**{"document__{0}_id__isnull".format(document_type): False})
    #                 #     return doc_topics.filter(topic__model=self).order_by("-value")[:n]
    #                 #
    #                 # def topic_distribution(self):
    #                 #
    #                 #     df = pandas.DataFrame(
    #                 #         list(DocumentTopic.objects.filter(topic__model=self).values("topic_id", "value"))
    #                 #     )
    #                 #     data = []
    #                 #     for index, row in df.groupby("topic_id").agg({"value": numpy.average}).sort("value").iterrows():
    #                 #         data.append((Topic.objects.get(pk=index), row["value"]))
    #                 #     return data


class Topic(LoggedExtendedModel):
    """
    A topic extracted by a topic model.
    """

    num = models.IntegerField()
    model = models.ForeignKey("django_learning.TopicModel", related_name="topics", null=True, on_delete=models.SET_NULL)
    name = models.CharField(max_length=50, db_index=True, null=True)
    label = models.CharField(max_length=300, db_index=True, null=True)
    anchors = ArrayField(models.CharField(max_length=100), default=[])

    class Meta:

        unique_together = ("num", "model")

    def __str__(self):

        if self.name:
            return self.name
        elif self.label:
            return self.label
        else:
            return self.top_ngrams()

    def set_label(self, label):

        self.label = label
        self.save()

    def anchor_string(self):

        return ", ".join(self.anchors)

    def top_ngrams(self, top_n=25):

        return " ".join([n.replace(" ", "_") for n in
                         self.ngrams.order_by("-weight").filter(weight__gte=.001).values_list("name", flat=True)[:top_n]])

    # def distinctive_ngrams(self):
    #
    #     return " ".join([ngram.name.replace(" ", "_") for ngram in self.ngrams.order_by("-weight") if
    #                      get_model("TopicNgram").objects.filter(name=ngram.name).filter(
    #                          topic__model=self.model).order_by("-weight")[0].topic == self])
    #
    # def most_similar(self, with_value=False):
    #     sim = self.similar_topics()
    #     if len(sim) > 0:
    #         if with_value:
    #             return sim[0]
    #         else:
    #             return sim[0][0]
    #     else:
    #         return None

    def values(self):
        return list(self.documents.values_list("value", flat=True))

    def values_at_least_1pct(self):
        return list(self.documents.filter(value__gte=.01).values_list("value", flat=True))

    def top_n_documents(self, n=10):
        return self.documents.order_by("-value")[:n]

    # def top_n_tweets(self, n=10):
    #     return self.tweets.order_by("-value")[:n]
    #
    # def top_n_bills(self, n=10):
    #     return self.bills.order_by("-value")[:n]

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
    topic = models.ForeignKey("django_learning.Topic", related_name="ngrams", on_delete=models.CASCADE)
    weight = models.FloatField()

    def __str__(self):
        return "{}*{}".format(self.name, self.weight)


class DocumentTopic(LoggedExtendedModel):
    topic = models.ForeignKey("django_learning.Topic", related_name="documents", on_delete=models.CASCADE)
    document = models.ForeignKey("django_learning.Document", related_name="topics", on_delete=models.CASCADE)
    value = models.FloatField()

    class Meta:
        unique_together = ("topic", "document")

    def __str__(self):
        return "{0}, {1}: {2}".format(
            str(self.document),
            str(self.topic),
            self.value
        )
