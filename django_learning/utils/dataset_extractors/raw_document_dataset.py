import pandas, inspect

from pewtils import is_not_null
from django_pewtils import get_model

from django_learning.utils.dataset_code_filters import dataset_code_filters
from django_learning.utils.dataset_document_filters import dataset_document_filters
from django_learning.utils.dataset_coder_filters import dataset_coder_filters
from django_learning.utils.balancing_variables import balancing_variables
from django_learning.utils.dataset_extractors import DatasetExtractor
from django_learning.utils.scoring import compute_scores_from_dataset, compute_overall_scores_from_dataset
from django_learning.functions import get_sampling_weights


class Extractor(DatasetExtractor):

    def __init__(self, **kwargs):

        document_ids = kwargs.get("document_ids", [])
        sampling_frame_name = kwargs.get("sampling_frame_name", None)
        document_filters = kwargs.get("document_filters", None)

        super(Extractor, self).__init__(**kwargs)

        self.documents = get_model("Document", app_name="django_learning").objects.all()

        self.sampling_frame = None
        if is_not_null(sampling_frame_name):
            self.sampling_frame = get_model("SamplingFrame", app_name="django_learning").objects.get(name=sampling_frame_name)
            self.documents = self.sampling_frame.documents.all()
        if is_not_null(document_ids, empty_lists_are_null=True):
            self.documents = self.documents.filter(pk__in=document_ids)

        self.document_filters = document_filters if document_filters else []

        self.index_levels = ["document_id"]
        self.outcome_column = None
        self.outcome_columns = None
        self.discrete_classes = None
        self.valid_label_ids = False

    def get_hash(self, **kwargs):

        hash_key = super(Extractor, self).get_hash(**kwargs)
        hash_key += str(inspect.getsourcelines(self._additional_steps))

        return self.cache.file_handler.get_key_hash(hash_key)

    def _test_index(self, dataset):

        if len(dataset) != len(dataset.groupby(self.index_levels).count()):
            raise Exception("All {} combinations must be unique!".format(self.index_levels))

    def _get_preserved_state(self, **kwargs):

        return {
            "outcome_column": self.outcome_column,
            "outcome_columns": self.outcome_columns,
            "discrete_classes": self.discrete_classes,
            "valid_label_ids": self.valid_label_ids
        }

    def _get_dataset(self, **kwargs):

        dataset = pandas.DataFrame.from_records(
            self.documents.values(
                "pk",
                "text",
                "date"
            )
        )
        dataset = dataset.rename(columns={"pk": "document_id"})

        if len(dataset) > 0:

            dataset = self._apply_filters(dataset)
            dataset = self._additional_steps(dataset, **kwargs)
            self._test_index(dataset)
            dataset["document_type"] = dataset["document_id"].map(lambda x: get_model("Document", app_name="django_learning").objects.get(pk=x).document_type)

        return dataset

    def _additional_steps(self, dataset, **kwargs):

        return dataset

    def _apply_filters(self, dataset):

        for filter_name, filter_args, filter_kwargs in self.document_filters:
            dataset = dataset_document_filters[filter_name](self, dataset, *filter_args, **filter_kwargs)

        return dataset