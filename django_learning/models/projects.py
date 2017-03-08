from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.contrib.auth.models import User

from pewtils import is_not_null
from pewtils.django import get_model

from django_learning.managers import QuestionManager
from django_learning.settings import DJANGO_LEARNING_BASE_MODEL, DJANGO_LEARNING_BASE_MANAGER
from django_learning.utils import projects, project_hit_types, project_qualification_tests, project_qualification_scorers


class Project(DJANGO_LEARNING_BASE_MODEL):

    name = models.CharField(max_length=250, unique=True)
    coders = models.ManyToManyField("django_learning.Coder", related_name="projects")
    admins = models.ManyToManyField("django_learning.Coder", related_name="admin_projects")
    blacklist = models.ManyToManyField("django_learning.Coder", related_name="blacklisted_projects")
    instructions = models.TextField(null=True)
    qualification_tests = models.ManyToManyField("django_learning.QualificationTest", related_name="projects")

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):

        if self.name not in projects.keys():
            raise Exception("Project '{}' is not defined in any of the known folders".format(self.name))

        config = projects[self.name]
        super(Project, self).save(*args, **kwargs)

        qual_tests = []
        for qual_test in config.get("qualification_tests", []):
            qual_tests.append(
                QualificationTest.objects.create_or_update({"name": qual_test})
            )
        self.qualification_tests = qual_tests

        for i, q in enumerate(config["questions"]):
            Question.objects.create_from_config("project", self, q, i)
        for c in config["coders"]:
            try: user = User.objects.get(username=c["name"])
            except User.DoesNotExist:
                user = User.objects.create_user(
                    c["name"],
                    "{}@pewresearch.org".format(c["name"]),
                    "pass"
                )
            coder = get_model("Coder").objects.create_or_update(
                {"name": c["name"]},
                {"is_mturk": False, "user": user}
            )
            self.coders.add(coder)
            if c["is_admin"]:
                self.admins.add(coder)

    def is_qualified(self, coder):

        return all([qual_test.is_qualified(coder) for qual_test in self.qualification_tests.all()])


class Question(DJANGO_LEARNING_BASE_MODEL):

    DISPLAY_CHOICES = (
        ('radio', 'radio'),
        ('checkbox', 'checkbox'),
        ('dropdown', 'dropdown'),
        ('text', 'text'),
        ('header', 'header')
    )

    qualification_test = models.ForeignKey("django_learning.QualificationTest", related_name="questions", null=True)
    project = models.ForeignKey("django_learning.Project", related_name="questions", null=True)

    name = models.CharField(max_length=250)
    prompt = models.TextField()
    display = models.CharField(max_length=20, choices=DISPLAY_CHOICES)
    multiple = models.BooleanField(default=False)
    tooltip = models.TextField(null=True)
    priority = models.IntegerField(default=1)

    objects = QuestionManager().as_manager()

    class Meta:
        unique_together = ("project", "qualification_test", "name")
        ordering = ['priority']

    def __str__(self):
        return "{}, {}".format(self.project, self.name)

    def labels_reversed(self):
        return self.labels.order_by("-priority")

    def update_assignment_response(self, assignment, label_values):

        existing = assignment.codes.filter(label__question=self)

        current = []
        if not self.multiple: labels = [label_values]
        else: labels = label_values

        if self.display == "number":
            labels = [
                Label.objects.create_or_update(
                    {"question": self, "value": l},
                    {"label": l}
                ) for l in labels
            ]
        else:
            labels = self.labels.filter(pk__in=labels)

        for l in labels:
            current.append(
                get_model("Code").objects.create_or_update(
                    {"assignment": assignment, "label": l}
                )
            )

        outdated = existing.exclude(current)
        outdated.delete()


    # def get_consensus_documents(self, label_value="1", turk_only=False, experts_only=False):
    #     return self.labels.get(value=label_value).get_consensus_documents(turk_only=turk_only, experts_only=experts_only)


class Label(DJANGO_LEARNING_BASE_MODEL):

    question = models.ForeignKey("django_learning.Question", related_name="labels")
    value = models.CharField(max_length=50, db_index=True, help_text="The code value")
    label = models.CharField(max_length=400, help_text="A longer label for the code value")
    priority = models.IntegerField(default=1)
    # TODO: reimplement: pointers = ArrayField(models.TextField(), default=[])
    select_as_default = models.BooleanField(default=False)

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    class Meta:

        unique_together = ("question", "value")
        ordering = ['priority']

    def __str__(self):
        return "{}: {}".format(self.question, self.label)

    # def get_consensus_documents(self, turk_only=False, experts_only=False):

    #     doc_ids = []
    #     sample_units = self.coder_documents.values_list("sample_unit_id", flat=True).distinct()
    #     for sid in sample_units:
    #         sample_unit = get_model("DocumentSampleDocument").objects.get(pk=sid)
    #         codes = sample_unit.codes.filter(code__variable=self.variable)
    #         if experts_only:
    #             codes = codes.filter(coder__is_mturk=False)
    #         if turk_only:
    #             codes = codes.filter(coder__is_mturk=True)
    #         codes = codes.exclude(consensus_ignore=True)
    #         codes = codes.values("code_id").distinct()
    #         if codes.count() == 1:
    #             doc_ids.append(sample_unit.document_id)

    #     return get_model("Document").objects.filter(pk__in=doc_ids)


class Example(DJANGO_LEARNING_BASE_MODEL):

    question = models.ForeignKey("django_learning.Question", related_name="examples")

    quote = models.TextField()
    explanation = models.TextField()

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()


class QualificationTest(DJANGO_LEARNING_BASE_MODEL):

    name = models.CharField(max_length=50, unique=True)
    coders = models.ManyToManyField("django_learning.Coder", related_name="qualification_tests", through="django_learning.QualificationAssignment")
    instructions = models.TextField(null=True)
    turk_id = models.CharField(max_length=250, unique=True, null=True)
    title = models.TextField(null=True)
    description = models.TextField(null=True)
    # TODO: reimplement: keywords = ArrayField(models.TextField(), default=[])
    price = models.FloatField(null=True)
    approval_wait_hours = models.IntegerField(null=True)
    duration_minutes = models.IntegerField(null=True)
    lifetime_days = models.IntegerField(null=True)

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):

        if self.name not in project_qualification_tests.keys():
            raise Exception("Qualification test '{}' is not defined in any of the known folders".format(self.name))

        config = project_qualification_tests[self.name]
        for attr in [
            "instructions",
            "title",
            "description",
            "price",
            "approval_wait_hours",
            "duration_minutes",
            "lifetime_days"
        ]:
            val = config.get(attr, None)
            setattr(self, attr, val)
        super(QualificationTest, self).save(*args, **kwargs)

        for i, q in enumerate(config["questions"]):
            Question.objects.create_from_config("qualification_test", self, q, i)

    def is_qualified(self, coder):

        try: return self.assignments.get(coder=coder).is_qualified
        except QualificationAssignment.DoesNotExist: return False


class QualificationAssignment(DJANGO_LEARNING_BASE_MODEL):

    test = models.ForeignKey("django_learning.QualificationTest", related_name="assignments")
    coder = models.ForeignKey("django_learning.Coder", related_name="qualification_assignments")

    time_started = models.DateTimeField(null=True, auto_now_add=True)
    time_finished = models.DateTimeField(null=True)
    turk_id = models.CharField(max_length=250, null=True)
    turk_status = models.CharField(max_length=40, null=True)

    is_qualified = models.NullBooleanField()
    # results = models.PickleFileField()

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    def save(self, *args, **kwargs):

        if is_not_null(self.time_finished):
            self.is_qualified = project_qualification_scorers[self.test.name](self)
        super(QualificationAssignment, self).save(*args, **kwargs)


class HITType(DJANGO_LEARNING_BASE_MODEL):

    project = models.ForeignKey("django_learning.Project", related_name="hit_types")
    name = models.CharField(max_length=50, unique=True)

    title = models.TextField(null=True)
    description = models.TextField(null=True)
    # TODO: reimplement: keywords = ArrayField(models.TextField(), default=[])
    price = models.FloatField(null=True)
    approval_wait_hours = models.IntegerField(null=True)
    duration_minutes = models.IntegerField(null=True)
    lifetime_days = models.IntegerField(null=True)
    min_approve_pct = models.FloatField(null=True)
    min_approve_cnt = models.IntegerField(null=True)

    turk_id = models.CharField(max_length=250, unique=True, null=True)

    qualification_tests = models.ManyToManyField("django_learning.QualificationTest", related_name="hit_types")

    objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()

    class Meta:
        unique_together = ("project", "name")

    def __str__(self):
        return "{}: {}".format(self.project, self.name)

    def save(self, *args, **kwargs):

        if self.name not in project_hit_types.keys():
            raise Exception("HIT Type '{}' is not defined in any of the known folders".format(self.name))

        config = project_hit_types[self.name]
        for attr in [
            "title",
            "description",
            "price",
            "approval_wait_hours",
            "duration_minutes",
            "lifetime_days",
            "min_approve_pct",
            "min_approve_cnt"
        ]:
            val = config.get(attr, None)
            setattr(self, attr, val)

        super(HITType, self).save(*args, **kwargs)

        qual_tests = []
        for qual_test in config.get("qualification_tests", []):
            qual_tests.append(
                QualificationTest.objects.create_or_update({"name": qual_test})
            )
        self.qualification_tests = qual_tests

    def is_qualified(self, coder):

        qual_tests = self.qualification_tests.all() | self.project.qualification_tests.all()
        return all([qual_test.is_qualified(coder) for qual_test in qual_tests])