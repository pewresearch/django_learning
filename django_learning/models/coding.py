from django.db import models
from django.db.models import Count
from django.contrib.auth.models import User
from django.contrib.postgres.fields import ArrayField

from django_commander.models import LoggedExtendedModel
from django_learning.managers import CodeManager
from django_learning.utils import project_hit_types


class HITType(LoggedExtendedModel):
    """
    Project HIT types specify a type of task to be completed. They primarily contain Mechanical Turk parameters
    specifying how much a task should cost, and other things like how long HITs should remain active until they expire.
    ``HITType`` objects in the database correspond to JSON config files.
    """

    project = models.ForeignKey(
        "django_learning.Project",
        related_name="hit_types",
        on_delete=models.CASCADE,
        help_text="Name of the project the HIT type belongs to",
    )
    name = models.CharField(
        max_length=50, help_text="Name of the HIT type (config file name)"
    )

    title = models.TextField(
        null=True, help_text="Short name to be given to the task to be performed"
    )
    description = models.TextField(null=True, help_text="More verbose description")
    keywords = ArrayField(
        models.TextField(),
        default=list,
        help_text="Search terms that Turkers can use to find the task",
    )
    price = models.FloatField(
        null=True, help_text="Price per HIT to be paid, in dollars"
    )
    approval_wait_hours = models.IntegerField(
        null=True,
        help_text="How many hours to wait before auto-approving completed tasks",
    )
    duration_minutes = models.IntegerField(
        null=True,
        help_text="Maximum number of minutes Turkers have to complete a single task",
    )
    lifetime_days = models.IntegerField(
        null=True,
        help_text="Maximum number of days uncompleted HITs will remain available after creation",
    )
    min_approve_pct = models.FloatField(
        null=True,
        help_text="Minimum approval percentage for workers to qualify for the HITs",
    )
    min_approve_cnt = models.IntegerField(
        null=True,
        help_text="Minimum number of good HITs workers must have done to qualify",
    )

    turk_id = models.CharField(
        max_length=250,
        unique=True,
        null=True,
        help_text="Unique Mechanical Turk ID, if it's been created via the API",
    )

    class Meta:
        unique_together = ("project", "name")

    def __str__(self):
        return "{}: {}".format(self.project, self.name)

    def save(self, *args, **kwargs):
        """
        Syncs the HIT type with its config file

        :param args:
        :param kwargs:
        :return:
        """

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
    """
    A HIT represents the combination of a SampleUnit and project codebook - a particular document to be coded by
    a certain number of coders. An ``Assignment`` will be created for each coder, linked to this HIT, when they begin
    coding. HITs can be in-house or on Mechanical Turk (the latter will have ``turk=True``), and when the
    specified ``num_coders`` have finished their assignments, the HIT will be marked ``finished=True``.
    """

    sample_unit = models.ForeignKey(
        "django_learning.SampleUnit",
        related_name="hits",
        on_delete=models.CASCADE,
        help_text="The SampleUnit the HIT is attached to; this is the document that will be coded",
    )
    hit_type = models.ForeignKey(
        "django_learning.HITType",
        related_name="hits",
        on_delete=models.SET_NULL,
        null=True,
        help_text="The HITType specifying things like how much the HIT pays",
    )
    template_name = models.CharField(
        max_length=250, null=True, help_text="(Optional) name of a custom HTML template"
    )
    num_coders = models.IntegerField(
        default=1, help_text="Number of coders to complete the HIT"
    )

    turk = models.BooleanField(
        default=False,
        help_text="Whether the HIT was deployed in-house (False) or on Mechanical Turk (True)",
    )
    turk_id = models.CharField(
        max_length=250,
        null=True,
        help_text="Unique HIT ID from the Mechanical Turk API (if applicable)",
    )
    turk_status = models.CharField(
        max_length=40,
        null=True,
        help_text="The HIT's status on Mechanical Turk (from the last time it was synced with the API)",
    )

    finished = models.BooleanField(
        null=True,
        help_text="Whether or not the HIT is complete (i.e. has been completed by ``num_coders`` coders",
    )

    # AUTO-FILLED RELATIONS
    sample = models.ForeignKey(
        "django_learning.Sample",
        related_name="hits",
        on_delete=models.CASCADE,
        help_text="The sample associated with the HIT (set automatically)",
    )

    def save(self, *args, **kwargs):
        """
        Extends the ``save`` methood to automaticaly mark the HIT as finished if it's associated with ``num_coders``
        completed assignments.
        :param args:
        :param kwargs:
        :return:
        """
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
    """
    Assignments are the combination of a HIT (task to be completed) and a particular coder. Assignments, in turn, are
    linked to the coder's actual coding decisions.
    """

    hit = models.ForeignKey(
        "django_learning.HIT",
        related_name="assignments",
        on_delete=models.CASCADE,
        help_text="The HIT the assignment belongs to",
    )
    coder = models.ForeignKey(
        "django_learning.Coder",
        related_name="assignments",
        on_delete=models.CASCADE,
        help_text="The coder that completed the assignment",
    )

    time_started = models.DateTimeField(
        null=True, auto_now_add=True, help_text="When the coder began coding"
    )
    time_finished = models.DateTimeField(
        null=True, help_text="When the coder finished coding"
    )

    turk_id = models.CharField(
        max_length=250,
        null=True,
        help_text="Unique Assignment ID from the Mechanical Turk ID (if applicable)",
    )
    turk_status = models.CharField(
        max_length=40,
        null=True,
        help_text="The assignment's status on Mechanical Turk (from the last time it was synced with the API)",
    )
    turk_approved = models.BooleanField(
        default=False,
        help_text="Whether or not the assignment's been approved and paid on Mechanical Turk",
    )

    notes = models.TextField(
        null=True,
        help_text="Any coder notes that were submitted alongside the responses",
    )
    uncodeable = models.BooleanField(
        default=False,
        help_text="Whether or not the coder marked the assignment as 'uncodeable' or too confusing to complete",
    )

    # AUTO-FILLED RELATIONS
    sample = models.ForeignKey(
        "django_learning.Sample",
        related_name="assignments",
        on_delete=models.CASCADE,
        help_text="The sample the assignment is associated with (set automatically)",
    )
    project = models.ForeignKey(
        "django_learning.Project",
        related_name="assignments",
        on_delete=models.CASCADE,
        help_text="The project the assignment is associated with (set automatically)",
    )

    def __str__(self):
        return "{}, {}".format(self.hit, self.coder)

    def save(self, *args, **kwargs):
        """
        Extendss the ``save`` function to auto-set the ``sample`` and ``project``.
        :param args:
        :param kwargs:
        :return:
        """

        self.sample = self.hit.sample
        self.project = self.hit.sample.project
        super(Assignment, self).save(*args, **kwargs)
        self.hit.save()


class Code(LoggedExtendedModel):
    """
    A code that was assigned to a particular sample unit / document during a coding assignment. This is linked to
    an assignment - which is then linked to a coder and a HIT, which is then linked to a sample unit - and a label
    that was selected.
    """

    label = models.ForeignKey(
        "django_learning.Label",
        related_name="codes",
        on_delete=models.CASCADE,
        help_text="A label that the coder selected",
    )
    assignment = models.ForeignKey(
        "django_learning.Assignment",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
        help_text="The assignment that was being coded",
    )
    qualification_assignment = models.ForeignKey(
        "django_learning.QualificationAssignment",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
        help_text="The qualification assignment that was being coded (if this is linked to a qualification test instead of a HIT)",
    )

    date_added = models.DateTimeField(
        auto_now_add=True, help_text="The date the code was created"
    )
    date_last_updated = models.DateTimeField(
        auto_now=True, help_text="The last date the code was modified"
    )

    consensus_ignore = models.BooleanField(
        default=False,
        help_text="(default is False) if True, this code will be ignored by document dataset extractors if ``exclude_consensus_ignore=True``. This flag gets marked when admins review and correct disagreements.",
    )

    notes = models.TextField(
        null=True,
        help_text="Notes from the coder, if code-level notes were enabled in the config",
    )

    # AUTO-FILLED RELATIONS
    coder = models.ForeignKey(
        "django_learning.Coder",
        related_name="codes",
        on_delete=models.CASCADE,
        help_text="The coder that completed the assignment (set automatically)",
    )
    hit = models.ForeignKey(
        "django_learning.HIT",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
        help_text="The HIT that was completed (set automatically)",
    )
    sample_unit = models.ForeignKey(
        "django_learning.SampleUnit",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
        help_text="The sample unit that was coded (set automatically)",
    )
    document = models.ForeignKey(
        "django_learning.Document",
        related_name="codes",
        null=True,
        on_delete=models.CASCADE,
        help_text="The document that was coded (set automatically)",
    )

    objects = CodeManager().as_manager()

    def __str__(self):
        return "{}: {}".format(
            self.assignment if self.assignment else self.qualification_assignment,
            self.label,
        )

    def save(self, *args, **kwargs):

        """
        Extends the ``save`` function to automatically set ``coder``, ``hit``, ``sample_unit`` and ``document``.
        :param args:
        :param kwargs:
        :return:
        """

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


class Coder(LoggedExtendedModel):

    """
    A coder that assigned codes to one or more documents, for one or more variables.  Non-MTurk coders are considered
    "experts".
    """

    name = models.CharField(
        max_length=200, unique=True, help_text="Unique name of the coder"
    )
    user = models.OneToOneField(
        User,
        related_name="coder",
        null=True,
        on_delete=models.SET_NULL,
        help_text="A Django auth User, if it's an in-house coder",
    )
    is_mturk = models.BooleanField(
        default=False, help_text="Whether or not the coder is a Mechanical Turk worker"
    )

    def __str__(self):
        return "{} ({})".format(self.name, "MTurk" if self.is_mturk else "In-House")

    def is_qualified(self, qual_test):
        """
        Returns whether the coder is qualified for the specified qualification test, based on the scorer function
        associated with the test.
        :param qual_test: A QualificationTest instance
        :return: True or False (whether they're qualified)
        """

        return qual_test.is_qualified(self)

    def _clear_abandoned_sample_assignments(self, sample):
        """
        Clears out unfinished assignments for the given coder for the specified sample
        :param sample: a Sample instance
        :return:
        """

        assignments = (
            self.assignments.filter(hit__sample=sample)
            .filter(time_finished__isnull=True)
            .annotate(c=Count("codes"))
            .filter(c=0)
        )

        assignments.delete()  # delete assignments that were accidentally abandoned
