Project setup
==============


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
