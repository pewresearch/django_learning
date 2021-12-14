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


class Question(LoggedExtendedModel):

    """
    Question objects specify a question to ask coders to fill out, and the coding options available to them. Questions
    can also be attached to qualification tests instead of projects. Question names should be unique within the
    specific project or qualification test they are attached to.
    """

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
        help_text="The qualification test the question belongs to",
    )
    project = models.ForeignKey(
        "django_learning.Project",
        related_name="questions",
        null=True,
        on_delete=models.SET_NULL,
        help_text="The project the question belongs to",
    )

    name = models.CharField(
        max_length=250,
        help_text="Short name of the question, must be unique to the project or qualification test",
    )
    prompt = models.TextField(help_text="The prompt that will be displayed to coders")
    display = models.CharField(
        max_length=20,
        choices=DISPLAY_CHOICES,
        help_text="The format of the question and how it will be displayed",
    )
    multiple = models.BooleanField(
        default=False, help_text="Whether or not multiple label selections are allowed"
    )
    tooltip = models.TextField(
        null=True,
        help_text="Optional text to be displayed when coders hover over the question",
    )
    priority = models.IntegerField(
        default=1,
        help_text="Order in which the question should be displayed relative to other questions. This gets set automatically based on the JSON config but can be modified manually. Lower numbers are higher priority.",
    )
    optional = models.BooleanField(
        default=False,
        help_text="(default is False) if True, coders will be able to skip the question",
    )
    show_notes = models.BooleanField(
        default=False,
        help_text="(default is False) if True, coders can write and submit notes about their decisions regarding this specific question",
    )

    dependency = models.ForeignKey(
        "django_learning.Label",
        related_name="dependencies",
        null=True,
        on_delete=models.SET_NULL,
        help_text="The label on another question that must be selected for this question to be displayed",
    )

    objects = QuestionManager().as_manager()

    class Meta:
        unique_together = ("project", "qualification_test", "name")
        ordering = ["priority"]

    def __str__(self):
        return "{}, {}".format(self.project, self.name)

    def labels_reversed(self):
        """
        Returns the questions label options in reverse priority order
        :return:
        """
        return self.labels.order_by("-priority")

    @property
    def has_pointers(self):
        """Whether or not any of the question labels have pointers associated with them"""
        if any(len(l.pointers) > 0 for l in self.labels.all()):
            return True
        else:
            return False

    def all_dependencies(self):
        """
        Recursively iterates through dependencies to return a query set of all Label options that must be selected
        for the question to be displayed
        :return:
        """

        label_ids = []
        dependency = self.dependency
        while dependency:
            label_ids.append(dependency.pk)
            dependency = dependency.question.dependency
        return get_model("Label", app_name="django_learning").objects.filter(
            pk__in=label_ids
        )

    def update_assignment_response(self, assignment, label_values, notes=None):
        """
        Updates the specified assignment with a list of label IDs
        :param assignment: Assignment instance that's being coded
        :param label_values: A list of label IDs that were selected (must belong to the question)
        :param notes: (Optional) notes that were passed along with the code
        :return:
        """

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
    """
    Labels represent response options to a question.
    """

    question = models.ForeignKey(
        "django_learning.Question",
        related_name="labels",
        on_delete=models.CASCADE,
        help_text="The question the label belongs to",
    )
    value = models.CharField(max_length=50, db_index=True, help_text="The code value")
    label = models.CharField(
        max_length=400, help_text="A longer label for the code value"
    )
    priority = models.IntegerField(
        default=1,
        help_text="Display priority relative to other label options, lower numbers are higher priority (default is 1)",
    )
    pointers = ArrayField(
        models.TextField(),
        default=list,
        help_text="List of bullet point-style tips for coders, specific to the particular label option",
    )
    select_as_default = models.BooleanField(
        default=False,
        help_text="(default is False) if True, this option will be selected as the default if no other option is chosen",
    )

    class Meta:

        unique_together = ("question", "value")
        ordering = ["priority"]

    def __str__(self):
        return "{}: {}".format(self.question, self.label)


class Example(LoggedExtendedModel):
    """
    An example given for a particular question, consisting of an example document (``quote``) and an ``explanation``
    that will be displayed in a pop-up modal if coders click on the question.
    """

    question = models.ForeignKey(
        "django_learning.Question",
        related_name="examples",
        on_delete=models.CASCADE,
        help_text="The question the example is assigned to",
    )

    quote = models.TextField(help_text="Example text")
    explanation = models.TextField(
        help_text="An explanation of how the text should be coded"
    )


class QualificationTest(LoggedExtendedModel):
    """
    Qualification tests provide sets of questions that coders must answer before they can qualify for coding on a
    particular project. Qualification tests can be reused across multiple projects. They're specified by JSON config
    files, and there must be a project qualification scorer function with the same name that can evaluate coders'
    responses and determine if they qualify.
    """

    name = models.CharField(
        max_length=50, help_text="Unique short name for the qualification test"
    )
    coders = models.ManyToManyField(
        "django_learning.Coder",
        related_name="qualification_tests",
        through="django_learning.QualificationAssignment",
        help_text="Coders that have taken the qualification test",
    )
    instructions = models.TextField(
        null=True, help_text="Instructions to be displayed at the top of the test"
    )
    turk_id = models.CharField(
        max_length=250,
        unique=True,
        null=True,
        help_text="Mechanical Turk ID for the test, if it's been synced via the API",
    )
    title = models.TextField(
        null=True, help_text="Title of the test (for Mechanical Turk)"
    )
    description = models.TextField(
        null=True, help_text="Description of the test (for Mechanical Turk)"
    )
    keywords = ArrayField(
        models.TextField(),
        default=list,
        help_text="List of keyword search terms (for Mechanical Turk)",
    )
    price = models.FloatField(
        null=True, help_text="How much to pay Mechanical Turk workers (in dollars)"
    )
    approval_wait_hours = models.IntegerField(
        null=True, help_text="How long to wait before auto-approving Turkers (in hours)"
    )
    duration_minutes = models.IntegerField(
        null=True, help_text="How long Turkers have to take the test (in minutes)"
    )
    lifetime_days = models.IntegerField(
        null=True, help_text="How long the test will be available (in days)"
    )
    mturk_sandbox = models.BooleanField(
        default=False,
        help_text="(default is False) if True, the test will be created in the Mechanical Turk sandbox",
    )

    def __str__(self):
        if self.sandbox:
            return "{} (MTURK SANDBOX)".format(self.name)
        else:
            return self.name

    class Meta:
        unique_together = ("name", "mturk_sandbox")

    def save(self, *args, **kwargs):
        """
        Extends the ``save`` function to sync the test with its JSON config with the same name
        :param args:
        :param kwargs:
        :return:
        """

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
        """
        Runs the qualification scorer to determine whether the specified coder passes the test
        :param coder: A Coder instance
        :return: True if the coder qualifies, False if not
        """

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
    """
    A particular coder's attempt at a qualification test.
    """

    test = models.ForeignKey(
        "django_learning.QualificationTest",
        related_name="assignments",
        on_delete=models.CASCADE,
        help_text="The qualification test the coder took",
    )
    coder = models.ForeignKey(
        "django_learning.Coder",
        related_name="qualification_assignments",
        on_delete=models.CASCADE,
        help_text="The coder that took the test",
    )

    time_started = models.DateTimeField(
        null=True, auto_now_add=True, help_text="When the coder started the test"
    )
    time_finished = models.DateTimeField(
        null=True, help_text="When the coder finished the test"
    )
    turk_id = models.CharField(
        max_length=250,
        null=True,
        help_text="The assignment ID from the Mechanical Turk API (if applicable)",
    )
    turk_status = models.CharField(
        max_length=40,
        null=True,
        help_text="The status of the assignment in Mechanical Turk (if applicable)",
    )
