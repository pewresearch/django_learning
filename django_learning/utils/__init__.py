import re, os

from collections import OrderedDict
from sklearn.pipeline import Pipeline, FeatureUnion
from django.apps import apps
from django.conf import settings

from pewtils import extract_attributes_from_folder_modules, extract_json_from_folder, decode_text
from pewtils.django import get_model, get_app_settings_folders


def get_document_types():

    # doc_types = set()
    # for app, model_list in apps.all_models.iteritems():
    #     for model_name, model in model_list.iteritems():
    #         if hasattr(model._meta, "is_document") and getattr(model._meta, "is_document"):
    #             doc_types.add(re.sub(" ", "_", model._meta.verbose_name))
    # return list(doc_types)
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
            val["name"] = params.func_name
        if function and hasattr(params, "params"):
            val["params"] = get_param_repr(params.params)
            return val
        else:
            return decode_text(params)


# for mod_category, attribute_name in [
#     ("balancing_variables", "var_mapper"),
#     ("code_filters", "filter"),
#     ("document_filters", "filter"),
#     ("feature_extractors", "Extractor"),
#     ("preprocessors", "Preprocessor"),
#     ("regex_filters", "get_regex"),
#     ("scoring_functions", "scorer"),
#     ("stopword_sets", "get_stopwords"),
#     ("pipelines", "get_pipeline"),
#     ("sampling_frames", "get_frame"),
#     ("sampling_methods", "get_method"),
#     ("project_qualification_scorers", "scorer"),
#     ("training_data_extractors", "get_training_data")
# ]:
#     mods = extract_attributes_from_folder_modules(
#         os.path.join(__path__[0], mod_category),
#         attribute_name,
#         include_subdirs=True,
#         concat_subdir_names=True
#     )
#     conf_var = "DJANGO_LEARNING_{}".format(mod_category.upper())
#     for folder in get_app_settings_folders(conf_var):
#         mods.update(
#             extract_attributes_from_folder_modules(
#                 folder,
#                 attribute_name,
#                 include_subdirs=True,
#                 concat_subdir_names=True
#             )
#         )
#     globals()[mod_category] = mods


# for json_category in [
#     "projects",
#     "project_hit_types",
#     "project_qualification_tests"
# ]:
#     mods = extract_json_from_folder(
#         os.path.join(__path__[0], json_category),
#         include_subdirs=True,
#         concat_subdir_names=True
#     )
#     conf_var = "DJANGO_LEARNING_{}".format(json_category.upper())
#     for folder in get_app_settings_folders(conf_var):
#         mods.update(
#             extract_json_from_folder(
#                 folder,
#                 include_subdirs=True,
#                 concat_subdir_names=True
#             )
#         )
#     globals()[json_category] = mods


# class LazyModule(dict):
#     def __init__(self, mod_category, attribute_name, *args, **kwargs):
#
#         self.mod_category = mod_category
#         self.attribute_name = attribute_name
#
#         self.attributes = None
#
#         super(LazyModule, self).__init__(*args, **kwargs)
#
#     def _load(self):
#
#         self.attributes = {}
#
#         mods = extract_attributes_from_folder_modules(
#             os.path.join(__path__[0], self.mod_category),
#             self.attribute_name
#         )
#         conf_var = "ADDITIONAL_{}".format(self.mod_category.upper())
#         if hasattr(settings, conf_var):
#             mods.update(
#                 extract_attributes_from_folder_modules(
#                     getattr(settings, conf_var),
#                     self.attribute_name
#                 )
#             )
#         self.attributes = mods
#
#     def __getitem__(self, item):
#         if not self.attributes: self._load()
#         return self.attributes[item]
#
#     def __repr__(self):
#         if not self.attributes: self._load()
#         return repr(self.attributes)
#
#     def __len__(self):
#         if not self.attributes: self._load()
#         return len(self.attributes)
#
#     def keys(self):
#         if not self.attributes: self._load()
#         return self.attributes.keys()
#
#     def items(self):
#         if not self.attributes: self._load()
#         return self.attributes.items()
#
#     def __contains__(self, item):
#         if not self.attributes: self._load()
#         return item in self.attributes
#
#     def __iter__(self):
#         if not self.attributes: self._load()
#         return iter(self.attributes)
#
# for mod_category, attribute_name in [
#     ("balancing_variables", "var_mapper"),
#     ("code_filters", "filter"),
#     ("document_filters", "filter"),
#     ("feature_extractors", "Extractor"),
#     ("preprocessors", "Preprocessor"),
#     ("regex_filters", "get_regex"),
#     ("scoring_functions", "scorer"),
#     ("stopword_sets", "get_stopwords")
# ]:
#     globals()[mod_category] = LazyModule(mod_category, attribute_name)
