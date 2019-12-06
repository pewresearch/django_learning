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



### Extracting datasets and collapsing coders
#### Components
- Dataset Code Filters
- Dataset Coder Filters
- Dataset Document Filters
- Dataset Extractors


## Machine learning

#### Components
- Balancing Variables
- Feature Extractors
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