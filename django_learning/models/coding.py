from django.db import models
from django.db.models import Count
from django.contrib.auth.models import User

from django_learning.settings import DJANGO_LEARNING_BASE_MODEL, DJANGO_LEARNING_BASE_MANAGER


class HIT(DJANGO_LEARNING_BASE_MODEL):

    sample_unit = models.ForeignKey("django_learning.SampleUnit", related_name="hits")
    template_name = models.CharField(max_length=250, null=True)
    num_coders = models.IntegerField(default=1)

    turk = models.BooleanField(default=False)
    turk_id = models.CharField(max_length=250, null=True)
    turk_status = models.CharField(max_length=40, null=True)

    # AUTO-FILLED RELATIONS
    sample = models.ForeignKey("django_learning.Sample", related_name="hits")

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    def save(self, *args, **kwargs):
        self.sample = self.sample_unit.sample
        super(HIT, self).save(*args, **kwargs)


class Assignment(DJANGO_LEARNING_BASE_MODEL):

    hit = models.ForeignKey("django_learning.HIT", related_name="assignments")
    coder = models.ForeignKey("django_learning.Coder", related_name="assignments")

    time_started = models.DateTimeField(null=True, auto_now_add=True)
    time_finished = models.DateTimeField(null=True)
    turk_id = models.CharField(max_length=250, null=True)
    turk_status = models.CharField(max_length=40, null=True)
    notes = models.TextField(null=True)
    uncodeable = models.BooleanField(default=False)

    # AUTO-FILLED RELATIONS
    sample = models.ForeignKey("django_learning.Sample", related_name="assignments")
    project = models.ForeignKey("django_learning.Project", related_name="assignments")

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    def __str__(self):
        return "{}, {}".format(self.hit, self.coder)

    def save(self, *args, **kwargs):

        self.sample = self.hit.sample
        self.project = self.hit.sample.project
        super(self, Assignment).save(*args, **kwargs)


class Code(DJANGO_LEARNING_BASE_MODEL):

    label = models.ForeignKey("django_learning.Label", related_name="codes")
    assignment = models.ForeignKey("django_learning.Assignment", related_name="codes")

    date_added = models.DateTimeField(auto_now_add=True, help_text="The date the document code was added")
    date_last_updated = models.DateTimeField(auto_now=True, help_text="The last date the document code was modified")

    consensus_ignore = models.BooleanField(default=False)

    # AUTO-FILLED RELATIONS
    coder = models.ForeignKey("django_learning.Coder", related_name="codes")
    hit = models.ForeignKey("django_learning.HIT", related_name="codes")
    sample_unit = models.ForeignKey("django_learning.SampleUnit", related_name="codes")
    document = models.ForeignKey("django_learning.Document", related_name="codes")

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    def __str__(self):
        return "{}: {}".format(self.assignment, self.label)

    def save(self, *args, **kwargs):

        self.coder = self.assignment.coder
        self.hit = self.assignment.hit
        self.sample_unit = self.assignment.hit.sample_unit
        self.document = self.assignment.hit.sample_unit.document
        super(self, Code).save(*args, **kwargs)

    # def save(self, *args, **kwargs):
    #        self.validate_unique()
    #        super(CoderDocumentCode, self).save(*args, **kwargs)

    #    def validate_unique(self, *args, **kwargs):
    #        super(CoderDocumentCode, self).validate_unique(*args, **kwargs)
    #        if not self.id:
    #            existing_codes = self.__class__.objects.filter(
    #                code__variable=self.code.variable,
    #                coder=self.coder,
    #                sample_unit=self.sample_unit
    #            )
    #            if existing_codes.count() > 0:
    #                raise ValidationError(
    #                    {
    #                        NON_FIELD_ERRORS: [
    #                            'CoderDocumentCode with the same variable already exists: {}'.format(
    #                                existing_codes.values_list("pk", flat=True)
    #                            )
    #                        ],
    #                    }
    #                )

    #    def __repr__(self):
    #        return "<CoderDocumentCode {0}, code__variable={1}, document_id={2}, hit_id={3}>".format(
    #            self.code.value, self.code.variable.name, self.sample_unit.document.id, self.hit_id
    #        )


class Coder(DJANGO_LEARNING_BASE_MODEL):

    """
    A coder that assigned codes to one or more documents, for one or more variables.  Non-MTurk coders are considered
    "experts".
    """

    name = models.CharField(max_length=200, unique=True, help_text="Unique name of the coder")
    user = models.OneToOneField(User, related_name="coder", null=True)
    is_mturk = models.BooleanField(default=False, help_text="Whether or not the coder is a Mechanical Turk worker")

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    def __repr__(self):
        return "<Coder {0}, is_turk={1}>".format(self.name, self.is_mturk)

    def is_qualified(self, qual_test):

        return qual_test.is_qualified(coder)

    # def is_qualified(self, project):
    #
    #     module = imp.load_source("limecoder", "project_files/{}/is_qualified.py".format(project.name))
    #     return module.test(self, project)

    def _clear_abandoned_sample_assignments(self, sample):

        assignments = self.assignments.filter(hit__sample=sample) \
            .filter(time_finished__isnull=True) \
            .annotate(c=Count("codes")) \
            .filter(c=0)

        assignments.delete()  # delete assignments that were accidentally abandoned