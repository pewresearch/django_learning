import importlib, re, os

from django.conf import settings

from sklearn.base import BaseEstimator, TransformerMixin

from pewtils import is_not_null, decode_text, extract_attributes_from_folder_modules, extract_json_from_folder
from pewtils.nlp import TextCleaner, SentenceTokenizer, is_probable_stopword
from django_pewtils import CacheHandler, get_model, get_app_settings_folders, reset_django_connection_wrapper

from django_learning.utils import get_param_repr
from django_learning.settings import LOCAL_CACHE_PATH



class BasicPreprocessor(object):

    def __init__(self, *args, **kwargs):

        self.params = kwargs
        self.param_repr = str(get_param_repr(self.params))
        if "cache_identifier" in self.params.keys() and is_not_null(self.params["cache_identifier"]):
            self.cache = CacheHandler(os.path.join(LOCAL_CACHE_PATH, "feature_extractors/{}/{}".format(self.params["cache_identifier"], self.name)), use_s3=False)
        else:
            self.cache = None

    def get_row_cache(self, key):

        if hasattr(self, "cache") and self.cache:
            key = decode_text(key)
            cache_key = str(self.name) + self.param_repr + str(key)
            return self.cache.read(cache_key)
            # return load_disk_cache(cache_key, folders=["learning", "feature_extractors", self.params['cache_identifier']], use_s3=False)
        else:
            return None

    def set_row_cache(self, key, data):

        if hasattr(self, "cache") and self.cache:
            key = decode_text(key)
            cache_key = str(self.name) + self.param_repr + str(key)
            self.cache.write(cache_key, data)
            # set_disk_cache(cache_key, data, folders=["learning", "feature_extractors", self.params['cache_identifier']], use_s3=False)


for mod_category, attribute_name in [
    ("preprocessors", "Preprocessor")
]:
    mods = extract_attributes_from_folder_modules(
        os.path.join(__path__[0]),
        attribute_name,
        include_subdirs=True,
        concat_subdir_names=True
    )
    conf_var = "DJANGO_LEARNING_{}".format(mod_category.upper())
    for folder in get_app_settings_folders(conf_var):
        mods.update(
            extract_attributes_from_folder_modules(
                folder,
                attribute_name,
                include_subdirs=True,
                concat_subdir_names=True
            )
        )
    globals()[mod_category] = mods
