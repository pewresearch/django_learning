import copy

from pewtils import is_null

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
        disable_probability_threshold_warning=False,
        **kwargs
    ):

        super(Extractor, self).__init__(**kwargs)

        self.dataset = copy.deepcopy(dataset)
        self.learning_model = learning_model
        self.cache_key = cache_key

        self.disable_probability_threshold_warning = disable_probability_threshold_warning

    def get_hash(self, **kwargs):

        hash_key = super(Extractor, self).get_hash(**kwargs)
        hash_key += self.learning_model.model_cache_hash + str(self.cache_key)

        return self.cache.file_handler.get_key_hash(hash_key)

    def _get_dataset(self, **kwargs):

        if is_null(self.learning_model.model):
            self.learning_model.load_model()
        dataset = self.dataset
        predictions = self.learning_model.apply_model(dataset, disable_probability_threshold_warning=self.disable_probability_threshold_warning)
        dataset[self.learning_model.dataset_extractor.outcome_column] = predictions[self.learning_model.dataset_extractor.outcome_column]
        dataset["probability"] = predictions["probability"]

        return dataset