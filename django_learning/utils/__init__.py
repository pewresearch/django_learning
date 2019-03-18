import re, os

from collections import OrderedDict
from sklearn.pipeline import Pipeline, FeatureUnion
from django.apps import apps
from django.conf import settings

from pewtils import extract_attributes_from_folder_modules, extract_json_from_folder, decode_text
from django_pewtils import get_model, get_app_settings_folders


def get_document_types():

    return [f.name for f in get_model("Document")._meta.get_fields() if f.one_to_one]


def get_pipeline_repr(pipeline, name=None):

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

    if type(params) == dict:
        new_params = OrderedDict()
        for k, v in sorted(params.items()):
            new_params[k] = get_param_repr(v)
        return new_params
    elif type(params) in [list, tuple]:
        return sorted([get_param_repr(p) for p in params])
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

    if "filter_dict" in params.keys() and params["filter_dict"]:
        objs = objs.filter(**params["filter_dict"])
    if "exclude_dict" in params.keys() and params["exclude_dict"]:
        objs = objs.exclude(**params["exclude_dict"])
    if "complex_filters" in params.keys() and params["complex_filters"]:
        for c in params["complex_filters"]:
            objs = objs.filter(c)

    return objs