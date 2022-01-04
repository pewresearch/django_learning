import re, os

import numpy as np

from collections import OrderedDict
from sklearn.pipeline import Pipeline, FeatureUnion
from django.apps import apps
from django.conf import settings

from pewtils import (
    extract_attributes_from_folder_modules,
    extract_json_from_folder,
    decode_text,
)
from django_pewtils import get_model, get_app_settings_folders


def get_document_types():
    """
    Returns a list of names of all models that have a one-to-one relationship with the Document model.
    :return:
    """

    return [f.name for f in get_model("Document")._meta.get_fields() if f.one_to_one]


def get_pipeline_repr(pipeline, name=None):
    """
    Given a machine learning pipeline configuration, returns a string representation that's used for hashing.
    :param pipeline:
    :param name:
    :return:
    """

    text_repr = OrderedDict()
    if isinstance(pipeline, Pipeline):
        text_repr["pipeline"] = []
        for sname, step in pipeline.steps:
            text_repr["pipeline"].append(get_pipeline_repr(step, name=sname))
    elif isinstance(pipeline, FeatureUnion):
        text_repr["feature_union"] = get_pipeline_repr(pipeline.transformer_list)
    elif isinstance(pipeline, tuple):
        text_repr.update(get_pipeline_repr(pipeline[1], name=pipeline[0]))
    elif isinstance(pipeline, list):
        text_repr["steps"] = []
        for sname, step in pipeline:
            text_repr["steps"].append(get_pipeline_repr(step, name=sname))
    else:
        text_repr["class"] = str(pipeline.__class__)
        text_repr["params"] = OrderedDict()
        if hasattr(pipeline, "get_params"):
            text_repr["params"] = get_param_repr(pipeline.get_params())
        if len(text_repr["params"].keys()) == 0:
            del text_repr["params"]

    return text_repr


def get_param_repr(params):
    """
    Given a dictionary of parameters, returns a unique string representation that's used for hashing.
    :param params:
    :return:
    """

    if type(params) == dict:
        new_params = OrderedDict()
        for k, v in sorted(params.items()):
            new_params[k] = get_param_repr(v)
        return new_params
    elif type(params) in [list, tuple]:
        repr = [get_param_repr(p) for p in params]
        try:
            repr = sorted(repr)
        except TypeError:
            pass
        return repr
    else:
        val = {}
        function = False
        if hasattr(params, "name"):
            function = True
            val["name"] = decode_text(params.name)
        elif hasattr(params, "func_name"):
            function = True
            val["name"] = params.__name__
        if function and hasattr(params, "params"):
            val["params"] = get_param_repr(params.params)
            return val
        else:
            return decode_text(params)


def filter_queryset_by_params(objs, params):
    """
    Takes a query set and a dictionary containing Django Learning sampling frame configuration parameters (see
    documentation) and filters the query set accordingly.
    :param objs: A QuerySet of objects
    :param params: A dictionary containing one or more of: filter_dict/exclude_dict/complex_filters/complex_excludes
    :return: A filtered query st
    """

    if "filter_dict" in params.keys() and params["filter_dict"]:
        objs = objs.filter(**params["filter_dict"])
    if "exclude_dict" in params.keys() and params["exclude_dict"]:
        objs = objs.exclude(**params["exclude_dict"])
    if "complex_filters" in params.keys() and params["complex_filters"]:
        for c in params["complex_filters"]:
            objs = objs.filter(c)
    if "complex_excludes" in params.keys() and params["complex_excludes"]:
        for c in params["complex_excludes"]:
            objs = objs.exclude(c)

    return objs


### DO NOT RELEASE
def wmom(arrin, weights_in, inputmean=None, calcerr=False, sdev=False):
    """
    A weighted average function that was taken from https://github.com/esheldon/esutil/blob/master/esutil/stat/util.py

    NAME:
      wmom()
    PURPOSE:
      Calculate the weighted mean, error, and optionally standard deviation of
      an input array.  By default error is calculated assuming the weights are
      1/err^2, but if you send calcerr=True this assumption is dropped and the
      error is determined from the weighted scatter.
    CALLING SEQUENCE:
     wmean,werr = wmom(arr, weights, inputmean=None, calcerr=False, sdev=False)
    INPUTS:
      arr: A numpy array or a sequence that can be converted.
      weights: A set of weights for each elements in array.
    OPTIONAL INPUTS:
      inputmean:
          An input mean value, around which them mean is calculated.
      calcerr=False:
          Calculate the weighted error.  By default the error is calculated as
          1/sqrt( weights.sum() ).  If calcerr=True it is calculated as sqrt(
          (w**2 * (arr-mean)**2).sum() )/weights.sum()
      sdev=False:
          If True, also return the weighted standard deviation as a third
          element in the tuple.
    OUTPUTS:
      wmean, werr: A tuple of the weighted mean and error. If sdev=True the
         tuple will also contain sdev: wmean,werr,wsdev
    REVISION HISTORY:
      Converted from IDL: 2006-10-23. Erin Sheldon, NYU
   """

    # no copy made if they are already arrays
    arr = np.array(arrin, ndmin=1, copy=False)

    # Weights is forced to be type double. All resulting calculations
    # will also be double
    weights = np.array(weights_in, ndmin=1, dtype="f8", copy=False)

    wtot = weights.sum()

    # user has input a mean value
    if inputmean is None:
        wmean = (weights * arr).sum() / wtot
    else:
        wmean = float(inputmean)

    # how should error be calculated?
    if calcerr:
        werr2 = (weights ** 2 * (arr - wmean) ** 2).sum()
        werr = np.sqrt(werr2) / wtot
    else:
        werr = 1.0 / np.sqrt(wtot)

    # should output include the weighted standard deviation?
    if sdev:
        wvar = (weights * (arr - wmean) ** 2).sum() / wtot
        wsdev = np.sqrt(wvar)
        return wmean, werr, wsdev
    else:
        return wmean, werr
