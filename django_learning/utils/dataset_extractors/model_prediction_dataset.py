import os, numpy, pandas, inspect, copy

from tqdm import tqdm

from django.conf import settings

from pewtils import classproperty, is_not_null, is_null, decode_text, extract_attributes_from_folder_modules, \
    extract_json_from_folder
from pewtils.django import CacheHandler, reset_django_connection_wrapper, get_model, get_app_settings_folders
from pewtils.sampling import compute_balanced_sample_weights

from django_learning.utils.dataset_code_filters import dataset_code_filters
from django_learning.utils.dataset_document_filters import dataset_document_filters
from django_learning.utils.dataset_coder_filters import dataset_coder_filters
from django_learning.utils.balancing_variables import balancing_variables
from django_learning.utils.dataset_extractors import DatasetExtractor


class Extractor(DatasetExtractor):

    def __init__(self,
        dataset=None,
        learning_model=None,
        cache_key=None,
        **kwargs
    ):

        super(ModelPredictionDatasetExtractor, self).__init__(**kwargs)

        self.dataset = dataset
        self.learning_model = learning_model
        self.cache_key = cache_key

    def get_hash(self, **kwargs):

        hash_key = super(ModelPredictionDatasetExtractor, self).get_hash(**kwargs)
        hash_key += self.cache_key + self.learning_model.model_hash

        return self.cache.file_handler.get_key_hash(hash_key)

    def _get_dataset(self, **kwargs):

        self.learning_model.load_model()
        dataset = self.dataset
        predictions = self.learning_model.apply_model(dataset)
        dataset[self.learning_model.outcome_variable] = predictions[self.learning_model.outcome_variable]
        dataset["probability"] = predictions["probability"]

        return dataset