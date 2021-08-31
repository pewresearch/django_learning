from django.db import models
from django.db.models import Count
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField

from django_commander.models import LoggedExtendedModel
from django_learning.managers import CodeManager
from django_learning.utils import project_hit_types


class HITType(LoggedExtendedModel):

    project = models.ForeignKey(
        "django_learning.Project", related_name="hit_types", on_delete=models.CASCADE
    )
    name = models.CharField(max_length=50)

    title = models.TextField(null=True)
    description = models.TextField(null=True)
    keywords = ArrayField(models.TextField(), default=list)
    price = models.FloatField(null=True)
    approval_wait_hours = models.IntegerField(null=True)
    duration_minutes = models.IntegerField(null=True)
    lifetime_days = models.IntegerField(null=True)
    min_approve_pct = models.FloatField(null=True)
    min_approve_cnt = models.IntegerField(null=True)

    turk_id = models.CharField(max_length=250, unique=True, null=True)

    class Meta:
        unique_together = ("project", "name")

    def __str__(self):
        return "{}: {}".format(self.project, self.name)

    def save(self, *args, **kwargs):

        if self.name not in project_hit_types.project_hit_types.keys():
            raise Exception(
                "HIT Type '{}' is not defined in any of the known folders".format(
                    self.name
                )
            )

        config = project_hit_types.project_hit_types[self.name]
        for attr in [
            "title",
            "description",
            "keywords",
            "price",
            "approval_wait_hours",
            "duration_minutes",
            "lifetime_days",
            "min_approve_pct",
            "min_approve_cnt",
        ]:
            val = config.get(attr, None)
            setattr(self, attr, val)

        super(HITType, self).save(*args, **kwargs)


class HIT(LoggedExtendedModel):

    sample_unit = models.ForeignKey(
        "django_learning.SampleUnit", related_name="hits", on_delete=models.CASCADE
    )
    hit_type = models.ForeignKey(
        "django_learning.HITType",
        related_name="hits",
        on_delete=models.SET_NULL,
        null=True,
    )
    template_name = models.CharField(max_length=250, null=True)
    num_coders = models.IntegerField(default=1)

    turk = models.BooleanField(default=False)
    turk_id = models.CharField(max_length=250, null=True)
    turk_status = models.CharField(max_length=40, null=True)

    finished = models.BooleanField(null=True)

    # AUTO-FILLED RELATIONS
    sample = models.ForeignKey(
        "django_learning.Sample", related_name="hits", on_delete=models.CASCADE
    )

    def save(self, *args, **kwargs):
        if (
            self.assignments.filter(time_finished__isnull=False).count()
            >= self.num_coders
        ):
            self.finished = True
        else:
            self.finished = False
        self.sample = self.sample_unit.sample
        super(HIT, self).save(*args, **kwargs)


class Assignment(LoggedExtendedModel):

    hit = models.ForeignKey(
        "django_learning.HIT", related_name="assignments", on_delete=models.CASCADE
    )
    coder = models.ForeignKey(
        "django_learning.Coder", related_name="assignments", on_delete=models.CASCADE
    )

    time_started = models.DateTimeField(null=True, auto_now_add=True)
    time_finished = models.DateTimeField(null=True)

    turk_id = models.CharField(max_length=250, null=True)
    turk_status = models.CharField(max_length=40, null=True)
    turk_approved = models.BooleanField(default=False)

    notes = models.TextField(null=True)
    uncodeable = models.BooleanField(default=False)

    # AUTO-FILLED RELATIONS
    sample = models.ForeignKey(
        "django_learning.Sample", related_name="assignments", on_delete=models.CASCADE
    )
    project = models.ForeignKey(
        "django_learning.Project", related_name="assignments", on_delete=models.CASCADE
    )

    def __str__(self):
        return "{}, {}".format(self.hit, self.coder)

    def save(self, *args, **kwargs):

        self.sample = self.hit.sample
        self.project = self.hit.sample.project
        super(Assignment, self).save(*args, **kwargs)
        self.hit.save()


class Code(LoggedExtendedModel):

    label = models.ForeignKey(
        "django_learning.Label", related_name="codes", on_delete=models.CASCADE
    )
    assignment = models.ForeignKey(
        "django_learning.Assignment",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
    )
    qualification_assignment = models.ForeignKey(
        "django_learning.QualificationAssignment",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
    )

    date_added = models.DateTimeField(
        auto_now_add=True, help_text="The date the document code was added"
    )
    date_last_updated = models.DateTimeField(
        auto_now=True, help_text="The last date the document code was modified"
    )

    consensus_ignore = models.BooleanField(default=False)

    notes = models.TextField(null=True)

    # AUTO-FILLED RELATIONS
    coder = models.ForeignKey(
        "django_learning.Coder", related_name="codes", on_delete=models.CASCADE
    )
    hit = models.ForeignKey(
        "django_learning.HIT", related_name="codes", null=True, on_delete=models.CASCADE
    )
    sample_unit = models.ForeignKey(
        "django_learning.SampleUnit",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
    )
    document = models.ForeignKey(
        "django_learning.Document",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
    )

    objects = CodeManager().as_manager()

    def __str__(self):
        return "{}: {}".format(
            self.assignment if self.assignment else self.qualification_assignment,
            self.label,
        )

    def save(self, *args, **kwargs):

        if self.assignment:
            self.coder = self.assignment.coder
            self.hit = self.assignment.hit
            self.sample_unit = self.assignment.hit.sample_unit
            self.document = self.assignment.hit.sample_unit.document
        elif self.qualification_assignment:
            self.coder = self.qualification_assignment.coder
            self.hit = None
            self.sample_unit = None
            self.document = None

        super(Code, self).save(*args, **kwargs)

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


class Coder(LoggedExtendedModel):

    """
    A coder that assigned codes to one or more documents, for one or more variables.  Non-MTurk coders are considered
    "experts".
    """

    name = models.CharField(
        max_length=200, unique=True, help_text="Unique name of the coder"
    )
    user = models.OneToOneField(
        User, related_name="coder", null=True, on_delete=models.SET_NULL
    )
    is_mturk = models.BooleanField(
        default=False, help_text="Whether or not the coder is a Mechanical Turk worker"
    )

    def __str__(self):
        return "{} ({})".format(self.name, "MTurk" if self.is_mturk else "In-House")

    def is_qualified(self, qual_test):

        return qual_test.is_qualified(self)

    def _clear_abandoned_sample_assignments(self, sample):

        assignments = (
            self.assignments.filter(hit__sample=sample)
            .filter(time_finished__isnull=True)
            .annotate(c=Count("codes"))
            .filter(c=0)
        )

        assignments.delete()  # delete assignments that were accidentally abandoned
