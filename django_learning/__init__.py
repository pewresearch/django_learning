import os

with open(os.path.join(os.path.dirname(__file__), "VERSION"), "rb") as version_file:
    __version__ = version_file.read().strip()

# print "Using django_learning from: {}".format(os.path.dirname(__file__))

import sklearn.metrics.scorer
from django_learning.sklearn_mods import _ProbaScorer
sklearn.metrics.scorer._ProbaScorer = _ProbaScorer

import sklearn.model_selection._validation
from django_learning.sklearn_mods import _fit_and_score
sklearn.model_selection._validation._fit_and_score = _fit_and_score
from django_learning.sklearn_mods import _score
sklearn.model_selection._validation._score = _score
from django_learning.sklearn_mods import _multimetric_score
sklearn.model_selection._validation._multimetric_score = _multimetric_score
sklearn.model_selection._validation._fit_and_score = _fit_and_score
import sklearn.model_selection._search
sklearn.model_selection._search._fit_and_score = _fit_and_score
