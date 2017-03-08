from django.db import models
# from django.contrib.postgres.fields import ArrayField

from django_learning.settings import DJANGO_LEARNING_BASE_MODEL, DJANGO_LEARNING_BASE_MANAGER
from django_learning.managers import NgramSetManager


class NgramSet(DJANGO_LEARNING_BASE_MODEL):

    name = models.CharField(max_length=100, db_index=True)
    dictionary = models.CharField(max_length=100, db_index=True)
    label = models.CharField(max_length=100, db_index=True, null=True)
    # TODO: reimplement
    # words = ArrayField(models.CharField(max_length=50), default=[])

    objects = NgramSetManager().as_manager()

    class Meta:
        unique_together = ("name", "dictionary")

    def __str__(self):
        name = self.label if self.label else self.name
        return "{0}: {1}".format(self.dictionary, name)


class DocumentNgramSet(DJANGO_LEARNING_BASE_MODEL):

    ngram_set = models.ForeignKey("django_learning.NgramSet", related_name="documents")
    document = models.ForeignKey("django_learning.Document", related_name="ngram_sets")
    count = models.IntegerField()
    percent = models.FloatField()

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    class Meta:
        unique_together = ("ngram_set", "document")

    def __str__(self):
        print "{0}, {1}: {2}".format(
            str(self.document),
            str(self.ngram_set),
            self.percent
        )

