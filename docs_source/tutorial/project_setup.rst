Creating a project (codebook)
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


Example project
-------------------------------


### JSON specifications

### Components
#### Projects
- Question types
- Dependencies
#### Project HIT Types
#### Project Qualification Scorers
#### Project Qualification Tests

### Commands
- create_project
- create_sample_hits
- extract_sample
- extract_sampling_frame
- create_coder

Starting a Project
==================

1. Create a project JSON file and put it in the projects folder (defined in ``settings.DJANGO_LEARNING_PROJECTS``, usually ``learning/projects``). This should contain the questions you want to ask.
2. Run the command ``python manage.py run_command create_project <name>`` where ``<name>`` is the JSON file's name **without the file extension**.
3. Every database object that you want to code must have a corresponding Document object to which it is related; make sure you've created Documents for all the codeable objects.
4. Now you need to define a sampling frame which filters the total universe of documents down to a subset with features you care about--this is the “population” of documents against which you’ll weight your sample. Sampling frames are python files with a ``def get_frame()`` that returns a dictionary of frame parameters, such as::

    def get_frame():
        return {
            'filter_dict': {
                'text__contains': 'pew'
            }
        }

5. Extract your sampling frame by running ``python manage.py run_command extract_sampling_frame <name>`` where ``<name>`` is the python file's name **without the file extension**.
6. Next, we need to create a JSON defining a HIT Type, which is mostly configuration for how mturk will list your tasks and who can do them--you will need a HIT type defined in ``settings.DJANGO_LEARNING_PROJECT_HIT_TYPES`` (usually ``learning/project_hit_types``) even for in-house / expert coding. An example hit type::

    {
        "display_type": "faces",
        "title": "Identify and label faces",
        "description": "We're identifying human faces in photos; labeling these faces accurately will help us train and validate machine learning models.",
        "keywords": ["face recognition", "images", "computer vision", "coding", "training"],
        "price": 0.05,
        "approval_wait_hours": 24,
        "duration_minutes": 10,
        "lifetime_days": 7,
        "min_approve_pct": 0.95,
        "min_approve_cnt": 100,
        "qualification_tests": []
    }

7. Now it’s time to extract a sample for coding. This tutorial will just use a random sampling process, which is the default. You will need to name the sample (underscores and alphanumeric characters only) and select a size::

    python manage.py run_command extract_sample <project name> <hit_type name> <sample name> --size <sample size> --sampling_frame_name <frame name>

8. Optionally, if you want to use a customized question layout rather than the default (which displays the ‘text’ field on each Document object), define that template in ``learning/project_hit_templates``.
