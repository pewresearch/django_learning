# Django Learning

Django Learning is a content analysis and machine learning framework designed to make it easy to 
develop a codebook, validate it, collect training data with crowdsourcing, and apply it at scale 
with machine learning.  At the moment, you won't find bleeding edge deep learning here; instead, 
this library is intended to provide a rigorous and reliable framework for developing ML models with 
tried-and-true methods and validating them to ensure that they achieve performance comparable to humans.

## Installation

### Dependencies

django_learning requires:

- Python (>= 2.7)
- Django (>= 1.10)
- Celery (>=4.0.2)
- [Pewtils (our own in-house Python utilities)](https://github.com/pewresearch/pewtils)
- [Django Pewtils (our own in-house Django utilities)](https://github.com/pewresearch/django_pewtils)
- [Django Commander (our own in-house command utilities)](https://github.com/pewresearch/django_commander)


### Project Components


### How Django Learning works

Django Learning is designed to be modular, allowing you to build a content analysis and machine learning 
pipeline out of isolated and reusable components that follow a standard syntax with predictable behavior. 
Everything, from project configuration files to machine learning feature extractors are stored in Python 
or JSON files within your project, and they are automatically detected by Django Learning when it 
initializes.  As you build your project files, you can organize them in a series of folders however 
you like, and tell Django Learning where to look using variables specified in `settings.py`. You can 
store files in a nested structure of subfolders, and the subfolder names will be concatenated into a 
unique name for each file as prefixes.  For example, if you wish to define a Django Learning Project - 
a series of questions to be asked about a set of documents - you might develop your project in a JSON 
file called `my_project.json` and put it in a folder in your Django app called `learning/projects`. 
To tell Django Learning where to look, you simply need to specify the location in `settings.py`:

```python
DJANGO_LEARNING_PROJECTS = [os.path.join(MY_APP_ROOT, "learning", "projects")]
```

Django Learning will then scan this - and any other folder specified by other compatible apps that you 
have installed - extract the JSON for any projects it detects, and make them all available via 
a single importable module:

```python
from django_learning.utils.projects import projects
projects['my_project']
```

## Documents - the basic unit in Django Learning

### Associating documents with your models


# Project (codebook) design

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


# Sampling

### Components
#### Sampling frames
#### Samples
#### Regex Filters
#### Sampling Frames
#### Sampling Methods


# Coding

## Coding in-house

## Coding on Mechanical Turk



# Extracting datasets and collapsing coders
## Components
### Dataset Code Filters
Code filters are the broadest way of excluding codes from a dataset when pulling an extract from the database. 
You can pass a parameter called `code_filters` to a `document_dataset`, `document_coder_datset`, or 
`document_coder_label` extractor, which should be a list of tuples of the form `(FILTER_NAME, args, kwargs)`, 
where `FILTER_NAME` corresponds to a file with a `filter` function defined in it, found in a folder contained 
in one of your `settings.DJANGO_LEARNING_DATASET_CODER_FILTERS` folders. The function should appear as follows:

```python
def filter(self, df, *args, **kwargs):

    return df[df["label_id"] == 12]
```
The function will receive the full dataframe of codes as the first argument (`df`), followed by any additional 
arguments and keyword arguments that you specified in a list and dictionary, respectively, when passing the name 
of the filter file to the extractor.

NOTE: code filters are always executed first, followed by coder filters, and then document filters.


### Dataset Coder Filters
Function the same way as code filters

### Dataset Document Filters
Function the same way as code/coder filters.

WITH ONE EXCEPTION: they will be also applied to the sampling frame when computing sampling weights. 
If you want to extract a dataset that's filtered in some way, and then weight it back to the full unfiltered sampling 
frame, you'll need to do that manually. Generally speaking, if you're systematically excluding certain observations 
from a sample, the subset shouldn't be used to make inferences about any data that was excluded. Right now, Django 
Learning assumes that if you're filtering to documents pertaining category or range (like dates), then those filters 
should be applied whenever the dataset is related back to the broader population from which it was drawn. This is 
particularly relevant when using a dataset to train a machine learning model; document filters will be propagated and 
used not only to compute sampling weights, but they will ALSO be automatically applied a trained model is applied to a 
dataset. Document filters are considered to be a universal scoping mechanism and they move in one direction only.

### Dataset Extractors


# Machine learning

## Components

#### Balancing Variables

Balancing variables can be used to specify additional partitions in your data that should be used in weighting. 
When extracting a `document_dataset`, `document_coder_datset`, or `document_coder_label` dataset, you can 
pass in a keyword argument containing the names of `balancing_variables` that correspond to functions defined 
in your `settings.DJANGO_LEARNING_BALANCING_VARIABLES` folders. Weights will be computed to evenly balance all 
classes within the partitions (using the combinations if multiple variables are included) and the results will 
be contained in a `balancing_weight` column in the returned dataset. If you specify a balancing weight in the 
training dataset for a machine learning pipeline, the `LearningModel` will use the balancing weight from the 
dataset when computing its training weight. If class weights and/or sampling weights are included, these will 
be multiplied by the balancing weight. Balancing weights will NOT be used when evaluating model performance. 
They exist in Django Learning solely for the training phase of machine learning, or simply for your own 
convenience when you're extracting datasets.

Example of a balancing variable file that balances documents evenly by month:

```python
from django_pewtils import get_model

def var_mapper(x):

    doc = get_model("Document", app_name="django_learning").objects.get(
        pk=x["document_id"]
    )
    if doc.date and doc.date.month and doc.date.year:
        return "{}_{}".format(doc.date.year, doc.date.month)
    else:
        return None
```

- Feature Extractors

All feature extractors extend traditional Scikit-Learn processors and expect to receive a dataframe that includes 
a `document_id` and `text` column. This allows for more sophisticated caching and feature extraction. 
- Models
- Pipelines
- Preprocessors
- Regex Replacers
- Scoring Functions
- Stopword Sets
- Stopword Whitelists=

#### Using a separate test/hold-out dataset for evaluation

#### Dealing with dependencies

Django Learning doesn't automatically impose dependencies because you may want to filter based on A) 
human labels (which might require a collapse rule) or B) the decisions of a model trained to predict 
the dependency.  Django Learning provides several Dataset Document Filters to specify this, respectively:

1) filter_by_existing_code
2) filter_by_other_model_dataset
3) filter_by_other_model_prediction


#### Topic modeling
- Topic Models


















***************
Django Learning
***************

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

In-House Coding
===============

1. Now that you have a sample extracted, you need to actually create coding assignments (HITs). For in-house coding, run::
    
    # do not include --template_name if using the default template
    python manage.py run_command create_sample_hits_experts <project name> <sample name> --num_coders <number coders per question> --template_name <custom HTML template filename, without extension>
    
2. Finally, you can access the sample coding interface via the url ``/learning/project/<project name>/sample/<sample name>``.

MTurk Coding
============

1. Now that you have a sample extracted, you need to actually create coding assignments (HITs) on mturk::
    
    # do not include --template_name if using the default template
    python manage.py run_command create_sample_hits_mturk <project name> <sample name> --prod --num_coders <number coders per question> --template_name <custom HTML template filename, without extension>
    
2. Until all hits are completed, you will need to pull down results periodically. You can additionally add ``--loop`` to sync every 30 seconds (unless a different time is supplied with ``--time_sleep``)::
    
    python manage.py run_command sync_sample_hits_mturk <project name> <sample name> --prod
    
3.  After syncing, you must also approve results so that the turkers get paid. It's wise to only approve a fraction of all complete-but-unapproved results at one time, so that turkers get the impression we are paying attention! Of course, you should  actually review some hits too!

    ::
    
        # probability is a 0-1.0 float, where e.g. .1 means 10% of 
        # complete-but-unapproved results will randomly be approved
        python manage.py run_command approve_sample_hits_mturk <project name> <sample name> --prod --probability .1
    
