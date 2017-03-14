import importlib

from sklearn.base import BaseEstimator, TransformerMixin

from pewtils import is_not_null, decode_text
from pewtils.django import CacheHandler, reset_django_connection_wrapper
from django_learning.utils import get_param_repr, preprocessors


class BasicExtractor(BaseEstimator, TransformerMixin):

    """
    Note: anything that's not serializable (like re.compile objects) cannot be set as an attribute at the time
    the model (and its extractors) is saved.  Instead, wrap them in functions that initialize them when they are
    needed, like get_preprocessors.

    http://scikit-learn.org/stable/developers/contributing.html#get-params-and-set-params

    """

    @reset_django_connection_wrapper
    def __init__(self, *args, **kwargs):

        self.params = {
            "document_types": None,
            "cache_identifier": None,
            "feature_name_prefix": None
        }
        self.param_repr = str(get_param_repr(self.params))
        self.cache = None
        self.set_params(*args, **kwargs)

    def get_params(self, *args, **kwargs):

        return self.params

    def set_params(self, *args, **kwargs):

        self.params.update(kwargs)
        super(BasicExtractor, self).set_params(*args, **kwargs)

        self.param_repr = str(get_param_repr(self.params))
        if is_not_null(self.params["cache_identifier"]):
            self.cache = CacheHandler("learning/feature_extractors/{}/{}".format(self.params["cache_identifier"], self.name), use_s3=False)

        return self

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

    def get_preprocessors(self):

        preprocessors = []
        if "preprocessors" in self.params.keys():
            for p, params in self.params["preprocessors"]:
                if is_not_null(self.params["cache_identifier"]):
                    params["cache_identifier"] = self.params["cache_identifier"]
                preprocessors.append(preprocessors[p](**params))

        return preprocessors