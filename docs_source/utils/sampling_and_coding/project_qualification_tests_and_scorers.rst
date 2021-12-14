Qualification tests and scorers
================================

Qualification tests are simplified versions project codebooks and HIT types combined into one. They're intended to
specify a series of questions that potential coders must answer before being allowed to complete any HITs. They are
intended to be paired with a qualification test scorer by the same name, which returns a True or False value indicating
whether the coder qualifies or not. To add a qualification test to a project config, simply add the name of it
to the qualification_tests section:

.. code:: python

    "qualification_tests": ["my_test"]

Qualification test config
--------------------------

Below is an example of a qualification test, which consists of a series of questions and some general Mechanical
Turk configuration parameters. For more information on
how to format the ``questions`` section of the config, you can refer to the Projects section of the documentation.
For more information on the other fields in the config, you can refer to the HIT Type section of the documentation.

.. code:: python

    {
      "instructions": "Here are some instructions to be displayed in the interface",
      "title": "Answer these questions to qualify",
      "description": "You need to answer correctly to qualify for the HITs",
      "price": 0.0,
      "approval_wait_hours": 24,
      "duration_minutes": 5,
      "lifetime_days": 7,
      "questions": [
        {
          "prompt": "Do you want to be qualified?",
          "name": "q1",
          "display": "radio",
          "labels": [
            {
              "label": "Yes",
              "value": "1"
            },
            {
              "label": "No",
              "value": "0"
            }
          ]
        }
      ]
    }

Qualification test JSON files should be placed in the ``settings.DJANGO_LEARNING_PROJECT_QUALIFICATION_TESTS`` folders.
Qualification tests function the same way as projects and HIT types; you can access the raw config via
``django_learning.utils``:

.. code:: python

    from django_learning.utils.project_qualification_tests import project_qualification_tests

    >>> project_qualification_tests["my_test"]()

    {
      "title": "Answer these questions to qualify",
      "description": "You need to answer correctly to qualify for the HITs",
      "keywords": ["labeling"],
      "price": 0.1,
      "approval_wait_hours": 24,
      "duration_minutes": 10,
      "lifetime_days": 7,
      "min_approve_pct": 0.95,
      "min_approve_cnt": 100
    }

However, qualification tests also get created in the database as ``QualificationTest`` objects, which sync with the
config file when you run ``.save()``:

.. code:: python

    from django_learning.models import QualificationTest

    >>> QualificationTest.objects.all()

    <BasicExtendedManager [<QualificationTest: my_test>]>

Qualification scorers
----------------------

Qualification scorers should be placed in the ``settings.DJANGO_LEARNING_PROJECT_QUALIFICATION_SCORERS`` folders. The
files should contain a single function called ``scorer`` that will be passed a ``QualificationTestAssignment`` object
corresponding to a coder's responses to the test. From this object, you can access all of their responses to different
questions. Ultimately, the function should return ``True`` if the coder qualifies, or ``False`` if they don't.

.. code:: python

    def scorer(qual_assignment):

        q1_code = qual_assignment.codes.get(label__question__name="q1")
        if int(q1_code.label.value) == 1:
            return True
        else:
            return False

Sandboxing
-----------

Qualification tests have separate versions stored for the Mechanical Turk sandbox. When you first create a project,
associated qualification tests will be created with ``mturk_sandbox=True``. When you switch the project out of the
sandbox, a new version of the qualification test will be created with ``mturk_sandbox=False`` if it doesn't already
exist. This allows you to preserve previously qualified coders if you've used the qualification tests before with the
live Mechanical Turk API.