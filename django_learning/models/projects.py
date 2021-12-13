from __future__ import print_function
from django.db import models
from django.contrib.postgres.fields import ArrayField
from django.contrib.auth.models import User
from django.contrib.contenttypes.fields import GenericRelation

from pewtils import is_not_null
from django_pewtils import get_model

from django_commander.models import LoggedExtendedModel
from django_learning.managers import QuestionManager
from django_learning.exceptions import RequiredResponseException
from django_learning.utils import projects
from django_learning.utils import dataset_extractors
from django_learning.utils import project_qualification_tests
from django_learning.utils import project_qualification_scorers

try:
    from importlib import reload

except ImportError:
    pass


class Project(LoggedExtendedModel):
    """
    Projects represent codebooks - sets of questions that you want to use to code a set of documents.

    """

    name = models.CharField(
        max_length=250,
        help_text="Name of the project (corresponds to a project config JSON file",
    )
    coders = models.ManyToManyField(
        "django_learning.Coder",
        related_name="projects",
        help_text="Coders on the project",
    )
    admins = models.ManyToManyField(
        "django_learning.Coder",
        related_name="admin_projects",
        help_text="Coders with admin privileges on the project",
    )
    instructions = models.TextField(
        null=True,
        help_text="Instructions to be displayed at the top of the coding interface",
    )
    qualification_tests = models.ManyToManyField(
        "django_learning.QualificationTest",
        related_name="projects",
        help_text="Qualification tests that coders must take to qualify for the coding project",
    )
    mturk_sandbox = models.BooleanField(
        default=True,
        help_text="(default is True) whether or not the project is in sandbox mode (for the purposes of interacting with Mechanical Turk)",
    )

    def __init__(self, *args, **kwargs):

        super(Project, self).__init__(*args, **kwargs)
        self.__init_mturk_sandbox = self.mturk_sandbox

    def __str__(self):
        if self.mturk_sandbox:
            return "{} (MTURK SANDBOX)".format(self.name)
        else:
            return self.name

    def save(self, *args, **kwargs):
        """
        Extends the ``save`` function to sync the project with its JSON config with the same name
        :param args:
        :param kwargs:
        :return:
        """

        if (
            self.mturk_sandbox == True
            and self.__init_mturk_sandbox == False
            and self.hits.filter(turk=True).count() > 0
        ):
            raise Exception(
                "This project already has live MTurk HITs, you can't switch it back to sandbox mode!"
            )
        elif self.mturk_sandbox == False and self.__init_mturk_sandbox == True:
            test_hits = project.hits.filter(turk=True)
            print(
                "About to delete {} sandbox HITs and switch to live mode: continue?".format(
                    test_hits.count()
                )
            )
            import pdb

            pdb.set_trace()
            test_hits.delete()
            from django_commander.commands import commands

            commands["django_learning_mturk_clear_sandbox"]()

        if self.name not in projects.projects.keys():
            reload(projects)
            if self.name not in projects.projects.keys():
                print(self.name)
                print(projects.projects.keys())
                import pdb

                pdb.set_trace()
                raise Exception(
                    "Project '{}' is not defined in any of the known folders".format(
                        self.name
                    )
                )

        config = projects.projects[self.name]
        if "instructions" in config.keys():
            self.instructions = config["instructions"]
        super(Project, self).save(*args, **kwargs)

        qual_tests = []
        for qual_test in config.get("qualification_tests", []):
            qual_tests.append(
                QualificationTest.objects.create_or_update(
                    {"name": qual_test, "mturk_sandbox": self.mturk_sandbox}
                )
            )
        self.qualification_tests.set(qual_tests)

        for i, q in enumerate(config["questions"]):
            Question.objects.create_from_config("project", self, q, i)

        admin_names = [c for c in config.get("admins", [])]
        coder_names = list(admin_names)
        coder_names.extend(config.get("coders", []))

        coders = []
        admins = []
        for c in coder_names:
            try:
                user = User.objects.get(username=c)
            except User.DoesNotExist:
                user = User.objects.create_user(
                    c, "{}@pewresearch.org".format(c), "pass"
                )
                # TODO: build in better user management
            coder = get_model("Coder").objects.create_or_update(
                {"name": c}, {"is_mturk": False, "user": user}
            )
            coders.append(coder.pk)
            if c in admin_names:
                admins.append(coder.pk)

        self.coders.set(get_model("Coder").objects.filter(pk__in=coders))
        self.admins.set(get_model("Coder").objects.filter(pk__in=admins))

    def expert_coders(self):
        """
        Returns in-house coders assigned to the project
        :return:
        """

        return self.coders.filter(is_mturk=False)

    def mturk_coders(self):
        """
        Returns Mechanical Turk coders assigned to the project
        :return:
        """

        return self.coders.filter(is_mturk=True)

    def is_qualified(self, coder):
        """
        Given a coder, returns whether or not they qualify based on the qualification tests associated with the project
        :param coder: a Coder instance
        :return:
        """

        return all(
            [
                qual_test.is_qualified(coder)
                for qual_test in self.qualification_tests.all()
            ]
        )

    def extract_document_coder_label_dataset(
        self, sample_names, question_names, code_filters=None, **kwargs
    ):

        e = dataset_extractors.dataset_extractors["document_coder_label_dataset"](
            project_name=self.name,
            sample_names=sample_names,
            question_names=question_names,
            **kwargs,
        )
        return e.extract(refresh=kwargs.get("refresh", False))

    def extract_document_coder_dataset(self, sample_names, question_names, **kwargs):

        e = dataset_extractors.dataset_extractors["document_coder_dataset"](
            project_name=self.name,
            sample_names=sample_names,
            question_names=question_names,
            **kwargs,
        )
        return e.extract(refresh=kwargs.get("refresh", False))

    def extract_document_dataset(self, sample_names, question_names, **kwargs):

        e = dataset_extractors.dataset_extractors["document_dataset"](
            project_name=self.name,
            sample_names=sample_names,
            question_names=question_names,
            **kwargs,
        )
        return e.extract(refresh=kwargs.get("refresh", False))


class Question(LoggedExtendedModel):

    DISPLAY_CHOICES = (
        ("radio", "radio"),
        ("checkbox", "checkbox"),
        ("dropdown", "dropdown"),
        ("text", "text"),
        ("header", "header"),
    )

    qualification_test = models.ForeignKey(
        "django_learning.QualificationTest",
        related_name="questions",
        null=True,
        on_delete=models.SET_NULL,
    )
    project = models.ForeignKey(
        "django_learning.Project",
        related_name="questions",
        null=True,
        on_delete=models.SET_NULL,
    )

    name = models.CharField(max_length=250)
    prompt = models.TextField()
    display = models.CharField(max_length=20, choices=DISPLAY_CHOICES)
    multiple = models.BooleanField(default=False)
    tooltip = models.TextField(null=True)
    priority = models.IntegerField(default=1)
    optional = models.BooleanField(default=False)
    show_notes = models.BooleanField(default=False)

    dependency = models.ForeignKey(
        "django_learning.Label",
        related_name="dependencies",
        null=True,
        on_delete=models.SET_NULL,
    )

    objects = QuestionManager().as_manager()

    class Meta:
        unique_together = ("project", "qualification_test", "name")
        ordering = ["priority"]

    def __str__(self):
        return "{}, {}".format(self.project, self.name)

    def labels_reversed(self):
        return self.labels.order_by("-priority")

    @property
    def has_pointers(self):
        if any(len(l.pointers) > 0 for l in self.labels.all()):
            return True
        else:
            return False

    def all_dependencies(self):

        label_ids = []
        dependency = self.dependency
        while dependency:
            label_ids.append(dependency.pk)
            dependency = dependency.question.dependency
        return get_model("Label", app_name="django_learning").objects.filter(
            pk__in=label_ids
        )

    def update_assignment_response(self, assignment, label_values, notes=None):

        existing = assignment.codes.filter(label__question=self)

        current = []
        if not self.multiple:
            labels = [label_values]
        else:
            labels = label_values
        labels = [l for l in labels if l]
        if self.display == "checkbox" and len(labels) == 0:
            labels = self.labels.filter(select_as_default=True)
            # if none of the other options were checked, choose the select_as_default option
        elif self.display == "number":
            labels = [
                Label.objects.create_or_update(
                    {"question": self, "value": l}, {"label": l}
                )
                for l in labels
            ]
            labels = self.labels.filter(pk__in=[l.pk for l in labels])
        elif self.display in ["text", "date"]:
            try:
                labels = self.labels.filter(value="open_response")
            except:
                label = Label.objects.create(question=self, value="open_response")
                self.labels.add(label)
                labels = self.labels.filter(value="open_response")
        else:
            labels = self.labels.filter(pk__in=[int(l) for l in labels])
        if labels.count() == 0 and not self.optional:
            raise RequiredResponseException()

        if "qualification" in assignment._meta.verbose_name:
            fk = "qualification_assignment"
        else:
            fk = "assignment"
        for l in labels:
            code = get_model("Code").objects.create_or_update(
                {fk: assignment, "label": l}
            )
            current.append(code.pk)

        outdated = existing.exclude(pk__in=current)
        outdated.delete()

        if is_not_null(notes):
            get_model("Code").objects.filter(pk__in=current).update(notes=notes)

    # def get_consensus_documents(self, label_value="1", turk_only=False, experts_only=False):
    #     return self.labels.get(value=label_value).get_consensus_documents(turk_only=turk_only, experts_only=experts_only)


class Label(LoggedExtendedModel):

    question = models.ForeignKey(
        "django_learning.Question", related_name="labels", on_delete=models.CASCADE
    )
    value = models.CharField(max_length=50, db_index=True, help_text="The code value")
    label = models.CharField(
        max_length=400, help_text="A longer label for the code value"
    )
    priority = models.IntegerField(default=1)
    pointers = ArrayField(models.TextField(), default=list)
    select_as_default = models.BooleanField(default=False)

    class Meta:

        unique_together = ("question", "value")
        ordering = ["priority"]

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


class Example(LoggedExtendedModel):

    question = models.ForeignKey(
        "django_learning.Question", related_name="examples", on_delete=models.CASCADE
    )

    quote = models.TextField()
    explanation = models.TextField()


class QualificationTest(LoggedExtendedModel):
    name = models.CharField(max_length=50)
    coders = models.ManyToManyField(
        "django_learning.Coder",
        related_name="qualification_tests",
        through="django_learning.QualificationAssignment",
    )
    instructions = models.TextField(null=True)
    turk_id = models.CharField(max_length=250, unique=True, null=True)
    title = models.TextField(null=True)
    description = models.TextField(null=True)
    keywords = ArrayField(models.TextField(), default=list)
    price = models.FloatField(null=True)
    approval_wait_hours = models.IntegerField(null=True)
    duration_minutes = models.IntegerField(null=True)
    lifetime_days = models.IntegerField(null=True)
    mturk_sandbox = models.BooleanField(default=False)

    def __str__(self):
        if self.sandbox:
            return "{} (MTURK SANDBOX)".format(self.name)
        else:
            return self.name

    class Meta:
        unique_together = ("name", "mturk_sandbox")

    def save(self, *args, **kwargs):

        if (
            self.name
            not in project_qualification_tests.project_qualification_tests.keys()
        ):
            raise Exception(
                "Qualification test '{}' is not defined in any of the known folders".format(
                    self.name
                )
            )

        config = project_qualification_tests.project_qualification_tests[self.name]
        for attr in [
            "instructions",
            "title",
            "description",
            "price",
            "approval_wait_hours",
            "duration_minutes",
            "lifetime_days",
        ]:
            val = config.get(attr, None)
            setattr(self, attr, val)
        super(QualificationTest, self).save(*args, **kwargs)

        for i, q in enumerate(config["questions"]):
            Question.objects.create_from_config("qualification_test", self, q, i)

    def is_qualified(self, coder):

        try:
            assignment = self.assignments.filter(time_finished__isnull=False).get(
                coder=coder
            )
            return project_qualification_scorers.project_qualification_scorers[
                self.name
            ](assignment)
        except QualificationAssignment.DoesNotExist:
            return False


class QualificationAssignment(LoggedExtendedModel):
    test = models.ForeignKey(
        "django_learning.QualificationTest",
        related_name="assignments",
        on_delete=models.CASCADE,
    )
    coder = models.ForeignKey(
        "django_learning.Coder",
        related_name="qualification_assignments",
        on_delete=models.CASCADE,
    )

    time_started = models.DateTimeField(null=True, auto_now_add=True)
    time_finished = models.DateTimeField(null=True)
    turk_id = models.CharField(max_length=250, null=True)
    turk_status = models.CharField(max_length=40, null=True)
