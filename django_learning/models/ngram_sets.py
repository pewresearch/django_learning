from __future__ import print_function
from django.db import models
from django.contrib.postgres.fields import ArrayField

from django_commander.models import LoggedExtendedModel
from django_learning.managers import NgramSetManager


class NgramSet(LoggedExtendedModel):
    """
    Represents a set of words that belong to a given category, e.g. a sentiment dicitonary. The category has a ``name``,
    and may belong to a specific ``dictionary`` (a broader set of categories). Name and dictionary should, together,
    be unique. Can optionally be given a more verbose ``label``, and the words that belong to the category exist
    in the ``words`` array field.
    """

    name = models.CharField(
        max_length=100, db_index=True, help_text="Short name of the category"
    )
    dictionary = models.CharField(
        max_length=100, db_index=True, help_text="Dictionary the category belongs to"
    )
    label = models.CharField(
        max_length=100,
        db_index=True,
        null=True,
        help_text="More verbose name for the category",
    )
    words = ArrayField(
        models.CharField(max_length=50),
        default=list,
        help_text="List of words in the category",
    )

    objects = NgramSetManager().as_manager()

    class Meta:
        unique_together = ("name", "dictionary")

    def __str__(self):
        name = self.label if self.label else self.name
        return "{0}: {1}".format(self.dictionary, name)


class DocumentNgramSet(LoggedExtendedModel):
    """
    An application of an ngram set on a specific document.
    """

    ngram_set = models.ForeignKey(
        "django_learning.NgramSet",
        related_name="documents",
        on_delete=models.CASCADE,
        help_text="The ngram set that was applied to the document",
    )
    document = models.ForeignKey(
        "django_learning.Document",
        related_name="ngram_sets",
        on_delete=models.CASCADE,
        help_text="The document the NgramSet was applied to",
    )
    count = models.IntegerField(help_text="Number of matches in the document")
    percent = models.FloatField(
        help_text="Proportion of the document that matched to the ngram set"
    )

    class Meta:
        unique_together = ("ngram_set", "document")

    def __str__(self):
        print(
            "{0}, {1}: {2}".format(
                str(self.document), str(self.ngram_set), self.percent
            )
        )
