from django.db import models
from django.contrib.postgres.fields import JSONField

from django_commander.models import LoggedExtendedModel

from pewtils import decode_text


class DocumentFragment(LoggedExtendedModel):
    """
    A chunk of content text that occurs in multiple documents of a certain scope (e.g. a footer with contact info
    that commonly occurs at the end of a particular politician's press releases, or a header that exists at the beginning
    of every news article from a particular media outlet.  Alternatively, could be a quote or entire article that's
    repeated in multiple places.  Boilerplate=True indicates non-content that should be removed, while Boilerplate=None
    indicates that the fragment has yet to be determined to be content or non-content.
    """

    scope = JSONField(default=dict,
                      help_text="A dictionary of filter parameters for defining documents within which the fragment can exist")

    documents = models.ManyToManyField("django_learning.Document", related_name="document_fragments")

    hash = models.CharField(max_length=256, db_index=True)
    text = models.TextField()
    # all_variations = ArrayField(models.TextField(), default=[])
    # boilerplate = models.NullBooleanField(null=True)

    def __str__(self):
        return "{0}...".format(decode_text(self.text)[:50])

