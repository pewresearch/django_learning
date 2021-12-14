Projects (codebooks)
-------------------------------

Projects are specified as JSON configuration files, and they define all of the questions you want to ask about
the documents you're coding. They have the following components:

Instructions
============

Stored in the "instructions" key, this can be whatever you want. Using the default coding interface, it'll be
displayed at the top.

Qualification tests
=======================

Stored in the "qualification_tests" key, this is a list of names of qualification tests that exist in one of
the ``settings.DJANGO_LEARNING_PROJECT_QUALIFICATION_TESTS`` folders you have. Coders will be required to
take and pass these tests before being allowed to code. If you don't have any tests, just put an empty list.

Admins
=======================

Stored in the "admins" key, this is a list of coder usernames who have admin privileges in the user interface.
Admins can add/remove coders, and can view and correct other coders' work.

Coders
=======================

Stored in the "coders" key, this is a list of coder usernames who will be coding on this project.

Questions
=======================

Questions comprise the bulk of your codebook. These are stored as a list of dictionaries in the "questions"
key in your project JSON. Questions have the following components:

    * "prompt": the prompt you want coders to see
    * "name": a short name for the variable being coded
    * "display": how the question should be displayed (options are "radio", "dropdown", "number", "checkbox", "text")
    * "labels": a list of coding options, which themselves are dictionaries; labels require a "label" (which is what is displayed), a "value" (which is what's stored in the database), and a list of "pointers" which are tips that appear in the popup. Checkbox questions can also support a single label with ``"select_as_default": true``
    * "tooltip": text that will be displayed when coders hover over the prompt
    * "examples": a list of dictionaries with "quote" and "explanation" attributes that appear in a popup when coders click on the prompt
    * "dependency": (optional) a dictionary containing a "question_name" and "label_value" indicating that question should only be displayed if another question is coded a certain way
    * "optional": if true, coders can skip the question
    * "show_notes": (optional) if true, coders will have the ability to add notes explaining their selection
    * "multiple": (for "dropdown" questions only) if true, allows coders to select multiple options


Radio questions
******************************

Radio questions get displayed as radio buttons; coders can choose only one option.

.. code:: json

    {
      "prompt": "Is this document good or bad?",
      "name": "good_bad",
      "display": "radio",
      "labels": [
        {
          "label": "Good",
          "value": "1",
          "pointers": ["Select this if it's good"]
        },
        {
          "label": "Bad",
          "value": "0",
          "pointers": ["Select this if it's bad"]
        }
      ],
      "tooltip": "Some additional details for coders",
      "examples": [
        {
          "quote": "This is a bad document",
          "explanation": "You should mark this as bad"
        }
      ]
    }

Dropdown questions
******************************

Dropdown questions are an alternative to radio questions. By default, coders can choose only one option.
However, if you add ``"multiple": true`` they will be able to select multiple codes.

.. code:: json

    {
      "prompt": "Which category fits this best?",
      "name": "category",
      "display": "dropdown",
      "multiple": false,
      "labels": [
        {
          "label": "Category A",
          "value": "a",
          "pointers": ["Category A is..."]
        },
        {
          "label": "Category B",
          "value": "b",
          "pointers": ["Category B is..."]
        }
      ],
      "tooltip": "Click for examples",
      "examples": [
        {
          "quote": "My example",
          "explanation": "My explanation"
        }
      ]
    }

Number questions
******************************

Number questions let coders enter an integer.

.. code:: json

    {
      "prompt": "How many cats do you see in this picture?",
      "name": "cats",
      "display": "number",
      "labels": [],
      "tooltip": "Count the cats",
      "examples": [
        {
          "quote": "This picture has one cat",
          "explanation": "Put the number one here"
        }
      ]
    }

Checkbox questions
******************************

Checkbox questions show a list of checkboxes. If the question is not optional, you must set one of your
labels to ``"select_as_default"==true``.

.. code:: json

    {
      "prompt": "Is this a thing?",
      "name": "thing",
      "display": "checkbox",
      "labels": [
        {
          "label": "Yes, it's a thing",
          "value": "thing",
          "pointers": []
        },
        {
          "label": "No",
          "value": "not_a_thing",
          "pointers": [],
          "select_as_default": true
        }
      ],
      "tooltip": "",
      "examples": [
        {
          "quote": "",
          "explanation": ""
        }
      ]
    }

Text questions (open-ends)
******************************

Text questions are open-ends, simple as that. Requires a placeholder label formatted like the example below.

.. code:: json

    {
      "prompt": "How does this document make you feel?",
      "name": "feelings",
      "display": "text",
      "labels": [{
          "label": "Open response",
          "value": "open_response",
          "pointers": [],
          "select_as_default": true
        }],
      "tooltip": "",
      "examples": [],
      "show_notes": true,
      "optional": true
    }

Syncing project config with the database
==============================================

Once you've specified a project and placed the JSON conig into one of the ``settings.DJANGO_LEARNING_PROJECTS``
folders, you can create the project in the database. You can do this either by running a command:

.. code:: python

    from django_commander.commands import commands
    commands["django_learning_coding_create_project"](project_name="my_project").run()

Or you can create the object in the database and tell it to sync with the config file by running ``.save()``:

.. code:: python

    from django_learning.models import Project
    project, _ = Project.objects.get_or_create(name="my_project")
    project.save()

Mechanical Turk sandboxing
==============================================

Projects (in the database) have ``mturk_sandbox`` set to ``True`` by default. Any qualification tests that are attached to them are
also created with ``mturk_sandbox=True``, allowing you to use the Mechanical Turk sandbox for testing. Once you're
satisfied with how your codebook functions, you can switch the project over to the live mode by running the
``django_learning_coding_mturk_exit_sandbox`` command:

.. code:: python

    from django_commander.commands import commands
    commands["django_learning_coding_mturk_exit_sandbox"](project_name="my_project").run()

Or you can do it from the command line: ``python manage.py run_command django_learning_coding_mturk_exit_sandbox my_project``.

This will automatically switch the project over to the live mode and will switch any qualification tests over to the
live versions (if live versions of the tests already exist, already-qualified coders will be pre-approved).