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
from pewanalytics.stats.sampling import SampleExtractor


class TopicModel(LoggedExtendedModel):
    """
    A topic model that's been fit to a specific sampling frame. The ``model`` and ``vectorizer`` are pickled and saved
    to their respective fields.
    """

    name = models.CharField(
        max_length=200, unique=True, help_text="Unique name for the topic model"
    )
    frame = models.ForeignKey(
        "django_learning.SamplingFrame",
        related_name="topic_models",
        on_delete=models.CASCADE,
        help_text="Sampling frame the topic model belongs to",
    )

    model = PickledObjectField(null=True, help_text="Pickled topic model")
    vectorizer = PickledObjectField(null=True, help_text="Pickled vectorizer")

    parameters = PickledObjectField(
        null=True, help_text="A pickle file of the parameters used"
    )

    training_documents = models.ManyToManyField(
        "django_learning.Document",
        related_name="topic_models_trained",
        help_text="Documents the model was trained on",
    )

    def __str__(self):

        return self.name

    def save(self, *args, **kwargs):
        """
        Extends the ``save`` function to pull and save the parameters from the config file
        :param args:
        :param kwargs:
        :return:
        """

        self.parameters = topic_models.topic_models[self.name]()
        self.frame = SamplingFrame.objects.get(name=self.parameters["frame"])
        super(TopicModel, self).save(*args, **kwargs)

    def _get_document_ids(self):
        """
        Returns the primary keys of the documents in the sampling frame
        :return:
        """

        doc_ids = list(
            self.frame.documents.filter(text__isnull=False).values_list("pk", flat=True)
        )

        return doc_ids

    def load_model(self, refresh_model=False, refresh_vectorizer=False):
        """
        Loads an existing model or trains a new model.
        :param refresh_model: (default is False) if True, the model will be re-trained even if it already exists
        :param refresh_vectorizer: (default is False) if True, the vectorizer will be re-fit even if it already exists
        :return:
        """

        if refresh_model or is_null(self.model):

            if refresh_vectorizer or is_null(self.vectorizer):

                from django_learning.utils.feature_extractors import feature_extractors

                self.vectorizer = feature_extractors["tfidf"](
                    **self.parameters["vectorizer"]
                )

                print("Extracting sample for vectorizer")

                frame_ids = self._get_document_ids()
                random.shuffle(frame_ids)

                if not self.parameters["sample_size"]:
                    sample_ids = frame_ids
                else:
                    sample_ids = SampleExtractor(
                        id_col="pk", sampling_strategy="random"
                    ).extract(
                        pandas.DataFrame({"pk": frame_ids}),
                        self.parameters["sample_size"],
                    )
                self.training_documents = get_model("Document").objects.filter(
                    pk__in=sample_ids
                )
                sample = pandas.DataFrame.from_records(
                    self.training_documents.values("pk", "text")
                )

                print("Training vectorizer on {} documents".format(len(sample)))
                self.vectorizer = self.vectorizer.fit(sample)
                print(
                    "{} features extracted from vectorizer".format(
                        len(self.vectorizer.get_feature_names())
                    )
                )

            print(
                "Initializing new topic model ({}, {})".format(
                    self.frame, self.parameters["num_topics"]
                )
            )
            self.model = corex_topic.Corex(
                n_hidden=self.parameters["num_topics"], seed=42
            )
            tfidf = self.vectorizer.transform(
                pandas.DataFrame.from_records(
                    self.training_documents.values("pk", "text")
                )
            )

            ngrams = self.vectorizer.get_feature_names()

            old_topics = list(self.topics.values())

            anchors = []
            anchor_topic_num = 0
            old_topic_map = {}
            for t in old_topics:
                if is_not_null(t["anchors"]):
                    anchor_list = [
                        anchor for anchor in t["anchors"] if anchor in ngrams
                    ]
                    if len(anchor_list) > 0:
                        anchors.append(anchor_list)
                        old_topic_map[anchor_topic_num] = t
                        anchor_topic_num += 1

            self.model = self.model.fit(
                tfidf,
                words=self.vectorizer.get_feature_names(),
                anchors=anchors,
                anchor_strength=self.parameters["anchor_strength"],
            )

            self.save()

            self.topics.all().delete()
            for i, topic_ngrams in enumerate(self.model.get_topics(n_words=20)):

                topic = Topic.objects.create_or_update(
                    {"num": i, "model": self},
                    search_nulls=False,
                    save_nulls=True,
                    empty_lists_are_null=True,
                )
                topic.ngrams.all().delete()
                for ngram, weight in topic_ngrams:
                    TopicNgram.objects.create(
                        name=str(ngram), topic=topic, weight=weight
                    )
                print(str(topic))

                old_topic = old_topic_map.get(i, None)
                if old_topic:

                    topic.name = old_topic["name"]
                    topic.label = old_topic["label"]
                    topic.anchors = old_topic["anchors"]
                    topic.save()

    def apply_model(self, df, probabilities=False):
        """
        Applies the topic model to a dataframe of documents with a ``text`` column.
        :param df: Dataframe with a ``text`` column
        :param probabilities: (default is False) if True, returns topic probabilities instead of discrete binary flags
        :return: A dataframe with additional columns for each topic in the model
        """

        tfidf = self.vectorizer.transform(df)
        topic_names = list(self.topics.order_by("num").values_list("name", flat=True))
        if not probabilities:
            topic_df = pandas.DataFrame(
                self.model.transform(tfidf), columns=topic_names
            ).astype(int)
        else:
            topic_df = pandas.DataFrame(
                self.model.transform(tfidf, details=True)[1], columns=topic_names
            ).astype(float)
        topic_df.index = df.index
        return pandas.concat([df, topic_df], axis=1)


class Topic(LoggedExtendedModel):
    """
    A topic extracted by a topic model.
    """

    num = models.IntegerField(
        help_text="Unique number of the topic in the model (unique together with ``model``)"
    )
    model = models.ForeignKey(
        "django_learning.TopicModel",
        related_name="topics",
        null=True,
        on_delete=models.SET_NULL,
        help_text="The model the topic belongs to (unique together with ``num``)",
    )
    name = models.CharField(
        max_length=50,
        db_index=True,
        null=True,
        help_text="Optional short name given to the topic",
    )
    label = models.CharField(
        max_length=300,
        db_index=True,
        null=True,
        help_text="Optional label/title given to the topic",
    )
    anchors = ArrayField(
        models.CharField(max_length=100),
        default=list,
        help_text="Optional list of terms to be used as anchors for this topic when training the model",
    )

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
        """
        Sets the topic's label
        :param label: Label to give to the topic
        :return:
        """

        self.label = label
        self.save()

    def anchor_string(self):
        """
        Returns a comma-delineated list of anchor terms belonging to the topic
        :return:
        """

        return ", ".join(self.anchors)

    def top_ngrams(self, top_n=25):
        """
        Returns a space-delineated string of the ``top_n`` terms belonging to the topic
        :param top_n: Number of top terms to return
        :return:
        """

        return " ".join(
            [
                n.replace(" ", "_")
                for n in self.ngrams.order_by("-weight")
                .filter(weight__gte=0.001)
                .values_list("name", flat=True)[:top_n]
            ]
        )


class TopicNgram(LoggedExtendedModel):
    """
    An ngram that belongs to a particular topic, with a given ``weight``
    """

    name = models.CharField(max_length=40, db_index=True, help_text="The ngram")
    topic = models.ForeignKey(
        "django_learning.Topic",
        related_name="ngrams",
        on_delete=models.CASCADE,
        help_text="The topic the ngram belongs to",
    )
    weight = models.FloatField(
        help_text="Weight indicating 'how much' the ngram belongs to the topic"
    )

    def __str__(self):
        return "{}*{}".format(self.name, self.weight)


class DocumentTopic(LoggedExtendedModel):
    """
    Application of a topic model to a document; the degree to which the topic matches the document is indicated by ``value``.
    """

    topic = models.ForeignKey(
        "django_learning.Topic",
        related_name="documents",
        on_delete=models.CASCADE,
        help_text="The topic",
    )
    document = models.ForeignKey(
        "django_learning.Document",
        related_name="topics",
        on_delete=models.CASCADE,
        help_text="The document",
    )
    value = models.FloatField(help_text="Presence of the topic in the document")

    class Meta:
        unique_together = ("topic", "document")

    def __str__(self):
        return "{0}, {1}: {2}".format(str(self.document), str(self.topic), self.value)
