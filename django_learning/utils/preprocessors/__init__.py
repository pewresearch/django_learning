import importlib

from sklearn.base import BaseEstimator, TransformerMixin

from pewtils import is_not_null, decode_text
from pewtils.nlp import TextCleaner, SentenceTokenizer, is_probable_stopword
from pewtils.django import CacheHandler
from pewtils.django import reset_django_connection_wrapper

from django_learning.utils import get_param_repr



class BasicPreprocessor(object):

    def __init__(self, *args, **kwargs):

        self.params = kwargs
        self.param_repr = str(get_param_repr(self.params))
        if "cache_identifier" in self.params.keys() and is_not_null(self.params["cache_identifier"]):
            self.cache = CacheHandler("learning/feature_extractors/{}/{}".format(self.params["cache_identifier"], self.name), use_s3=False)
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