# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from django_commander.models import LoggedExtendedModel
from django_learning.managers import DocumentManager
from django.apps import apps
from django.contrib.postgres.fields import ArrayField
from django.db import models
from django.db.models.signals import class_prepared
from langdetect import detect
from pewtils import is_not_null, is_null, decode_text
from pewtils import get_hash
import re


class Document(LoggedExtendedModel):

    """
    Documents are at the core of Django Learning - these are the pieces of text that you want to code. They also
    link all of the other models in Django Learning with the models in your own Django app. To do this, you simply
    need to create a ``OneToOneRelation`` with ``django_learning.models.Document`` with a model in your own app.
    For example, if you have a model for Facebook posts, you can create a Document for each post and populate the
    ``text`` field with the text of each post.
    """

    text = models.TextField(help_text="The text content of the document")
    date = models.DateTimeField(
        null=True, help_text="An optional date associated with the document"
    )

    ssdeep = models.CharField(
        max_length=256,
        null=True,
        db_index=True,
        help_text="Locally-sensitive ssdeep hash of the document (set automatically)",
    )

    language = models.CharField(
        max_length=5,
        null=True,
        help_text="If null (default), this gets auto-detected by ``langdetect``",
    )

    coded_labels = models.ManyToManyField(
        "django_learning.Label",
        related_name="coded_documents",
        through="django_learning.Code",
        help_text="Labels that have been assigned to the document by a coder",
    )
    classified_labels = models.ManyToManyField(
        "django_learning.Label",
        related_name="classified_documents",
        through="django_learning.Classification",
        help_text="Labels that have been assigned to the document by a classification model",
    )

    freeze_text = models.BooleanField(
        default=False,
        help_text="If True, the document text will be modified and attempts to change it will raise an exception",
    )

    objects = DocumentManager().as_manager()

    __init_text = None

    def __init__(self, *args, **kwargs):

        super(Document, self).__init__(*args, **kwargs)
        self.__init_text = self.text

    @classmethod
    def get_parent_relations(cls):
        """
        Returns models that have a one-to-one relationship with the Document table
        :return:
        """
        obj_models = []
        from django.db.models.fields.reverse_related import OneToOneRel

        for field in cls._meta.get_fields():
            if type(field) == OneToOneRel:
                obj_models.append(field)
        return obj_models

    @property
    def document_type(self):
        """
        Returns the name of the model that has a one-to-one relationship with this specific document
        :return:
        """

        for doc_type in self._meta.model.objects.document_types():
            if hasattr(self, doc_type) and getattr(self, doc_type):
                return doc_type
        return None

    @property
    def object(self):

        """
        Returns the "parent" object that has a one-to-one relationship with this document
        :return:
        """

        for parent_field in self.get_parent_relations():
            if hasattr(self, parent_field.name):
                return getattr(self, parent_field.name)
        return None

    def __str__(self):

        return "Document #{}, {}: {}...".format(
            self.pk, str(self.object), decode_text(self.text[:50])
        )

    def freeze(self):
        """Sets ``freeze_text=True`` and freezes the document text"""

        self.freeze_text = True
        self.save()

    def unfreeze(self):
        """ Sets ``freeze_text=False`` and unfreezes the document text"""

        self.freeze_text = False
        self.save()

    def save(self, *args, **kwargs):
        """
        Extends the ``save`` function and detects changes to the text. If changes have occurred, a warning and
        confirmation prompt will be raised by default before existing text-dependent relations are cleared out.

        :param args:
        :param ignore_warnings: (default is False) if True, ignores warnings about modified text and will automatically
            delete existing labels and other text-dependent relations if the text has been modified
        :param allow_modified_text: (default is False) if True, existing text-dependent relations will be preserved
            even if the text has been modified
        :param kwargs:
        :return:
        """

        if "ignore_warnings" in kwargs.keys():
            ignore_warnings = kwargs.pop("ignore_warnings")
        else:
            ignore_warnings = False

        if "allow_modified_text" in kwargs.keys():
            allow_modified_text = kwargs.pop("allow_modified_text")
        else:
            allow_modified_text = False

        if not self.language:
            try:
                self.language = detect(self.text)
            except:
                self.language = "unk"

        if not allow_modified_text:
            if self.__init_text != self.text and not self.freeze_text:
                self.ssdeep = get_hash(self.text, hash_function="ssdeep")
                for m2m in [
                    "ngram_sets",
                    "topics",
                    "classified_labels",
                    "coded_labels",
                ]:
                    if getattr(self, m2m).count() > 0:
                        delete = True
                        if not ignore_warnings:
                            print(
                                "Warning: text for document {} was modified, continuing will clear out {}".format(
                                    self.pk, m2m
                                )
                            )
                            print(
                                "Quit, or set delete=False and continue to skip this from happening"
                            )
                            print(
                                "To allow text modifications to happen without triggering this warning, pass allow_modified_text=True to the Document save function"
                            )
                            import pdb

                            pdb.set_trace()
                        if delete:
                            if hasattr(self, m2m) and hasattr(
                                hasattr(self, m2m), "through"
                            ):
                                getattr(self, m2m).through.delete()
                            else:
                                try:
                                    getattr(self, m2m).clear()
                                except:
                                    getattr(self, m2m).all().delete()
                        else:
                            print("Manual override, skipping")
            else:
                self.text = self.__init_text

        super(Document, self).save(*args, **kwargs)
        if self.object:
            # For some really stupid reason, Django doesn't save one-to-one relationships from the reverse model
            self.object.document = self
            self.object.save()

        self.__init_text = self.text
