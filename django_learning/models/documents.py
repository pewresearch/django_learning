import re

from django.db import models
from django.contrib.postgres.fields import ArrayField, JSONField
from django.apps import apps
from django.db.models.signals import class_prepared

from langdetect import detect

from pewtils import is_not_null, is_null, decode_text
from pewtils.nlp import get_hash
from pewtils.django import get_fields_with_model

from django_commander.models import LoggedExtendedModel
from django_learning.managers import DocumentManager
from django_queries.models import QueryModel


class Document(LoggedExtendedModel, QueryModel):

    text = models.TextField(help_text="The text content of the document")
    original_text = models.TextField(null=True)
    duplicate_ids = ArrayField(models.IntegerField(), default=[])
    alternative_text = ArrayField(models.TextField(), default=[])
    date = models.DateTimeField(null=True, help_text="An optional date associated with the document")

    is_clean = models.BooleanField(default=False)

    # nilsimsa = models.CharField(max_length=256, null=True, db_index=True)
    # tlsh = models.CharField(max_length=256, null=True, db_index=True)
    ssdeep = models.CharField(max_length=256, null=True, db_index=True)

    language = models.CharField(max_length=5, null=True)

    paragraphs = models.ManyToManyField("self", symmetrical=False, related_name="parent")
    paragraph_id = models.IntegerField(null=True)

    entities = models.ManyToManyField("django_learning.Entity", related_name="documents")

    coded_labels = models.ManyToManyField("django_learning.Label",
        related_name="coded_documents",
        through="django_learning.Code"
    )
    # classified_labels = models.ManyToManyField("django_learning.Label",
    #     related_name="classified_documents",
    #     through="django_learning.Classification"
    # )

    freeze_text = models.BooleanField(default=False)

    external_link = models.URLField(null=True, max_length=500)

    objects = DocumentManager().as_manager()

    __init_text = None

    def __init__(self, *args, **kwargs):

        super(Document, self).__init__(*args, **kwargs)
        self.__init_text = self.text

    @classmethod
    def get_parent_relations(cls):
        obj_models = []
        from django.db.models.fields.reverse_related import OneToOneRel
        for field in cls._meta.get_fields():
            if type(field) == OneToOneRel:
                obj_models.append(field)
        return obj_models

    @property
    def object(self):

        for parent_field in self.get_parent_relations():
            if hasattr(self, parent_field.name):
                return getattr(self, parent_field.name)
        return None

        # from django.db.models.fields.reverse_related import OneToOneRel
        # for field, _ in get_fields_with_model(self):
        #     if type(field) == OneToOneRel and hasattr(self, field.name):
        #         return getattr(self, field.name)
        # return None

    def __str__(self):

        return "Document #{}, {}: {}...".format(
            self.pk,
            str(self.object),
            decode_text(self.text[:50])
        )

    #     # parent_field = self.get_parent_field()
    #     # if parent_field:
    #     #     parent_str = str(parent_field.related_model.objects.get(pk=getattr(self, parent_field.name).pk))
    #     #     return "Document #{}, <{}>: {}...".format(
    #     #         self.pk,
    #     #         parent_str,
    #     #         decode_text(self.text[:50])
    #     #     )
    #     #     # return "Document, {0}_id={1}: {2}{3}".format(
    #     #     #     str(parent_field.related_model._meta.model_name),
    #     #     #     getattr(self, parent_field.name).pk,
    #     #     #     decode_text(self.text[:50]),
    #     #     #     "..."
    #     #     # )
    #     # else:
    #     #     return "Document #{}, <no parent>: {}...".format(
    #     #         self.pk,
    #     #         decode_text(self.text[:50])
    #     #     )

    def _update_paragraphs(self):

        if not self.parent:
            for i, paragraph in enumerate(self.text.split("\n")):
                self.paragraphs.create_or_update(
                    {"paragraph_id": i},
                    {
                        "text": paragraph,
                        "date": self.date,
                        "is_clean": self.is_clean,
                        "language": self.language
                    },
                    return_object=False
                )

    def freeze(self):

        self.freeze_text = True
        self.save()

    def unfreeze(self):

        self.freeze_text = False
        self.save()

    def save(self, *args, **kwargs):

        if "ignore_warnings" in kwargs.keys():
            ignore = kwargs.pop("ignore_warnings")
        else:
            ignore = False

        if not self.original_text:
            self.original_text = self.text

        if not self.language:
            try:
                self.language = detect(self.text)
            except:
                self.language = "unk"

        # self.nilsimsa = get_hash(self.text, hash_function="nilsimsa")
        # self.tlsh = get_hash(self.text, hash_function="tlsh")
        if self.__init_text != self.text and not self.freeze_text:
            self.ssdeep = get_hash(self.text, hash_function="ssdeep")
            for m2m in ["document_fragments", "entities", "ngram_sets", "topics"]:
                if getattr(self, m2m).count() > 0:
                    if not ignore:
                        print "Warning: text for document {} was modified, clearing out {}".format(self.pk, m2m)
                    # setattr(self, m2m, [])
                    # getattr(self, m2m).clear()
                    try:
                        getattr(self, m2m).clear()
                    except:
                        getattr(self, m2m).all().delete()
            # for su in self.sample_weights.all():
            #     # if not ignore:
            if self.coded_labels.count() > 0:
                print "Warning: text for document {} was modified, clearing out coded labels".format(self.pk)
                import pdb
                pdb.set_trace()
                self.codes.all().delete()
            # TODO: reenable
            # if self.classified_labels.count() > 0:
            #     print "Warning: text for document {} was modified, clearing out classified labels".format(self.pk)
            #     import pdb
            #     pdb.set_trace()
            #     self.classified_labels.through.delete()
            self._update_paragraphs()
        else:
            self.text = self.__init_text

        super(Document, self).save(*args, **kwargs)
        if self.object:
            # For some really stupid reason, Django doesn't save one-to-one relationships from the reverse model
            self.object.document = self
            self.object.save()

        self.__init_text = self.text
        if self.paragraphs.count() == 0:
            self._update_paragraphs()

    def get_parent_field(self):

        parent_field = None
        for f in self._meta.get_fields():
            if f.is_relation and f.one_to_one and hasattr(self, f.name) and is_not_null(getattr(self, f.name)):
                parent_field = f
        return parent_field


# def add_foreign_keys(sender, **kwargs):
#
#     if sender.__base__ == DocumentModel:
#         for app, model_list in apps.all_models.iteritems():
#             for model_name, model in model_list.iteritems():
#                 if hasattr(model._meta, "is_document") and getattr(model._meta, "is_document"):
#                     field = models.OneToOneField("{}.{}".format(app, model.__name__), null=True, related_name="document")
#                     field.contribute_to_class(sender, re.sub(" ", "_", model._meta.verbose_name))
#
# class_prepared.connect(add_foreign_keys)

