import os, numpy, pandas, inspect

from tqdm import tqdm

from django.conf import settings

from pewtils import classproperty, is_not_null, is_null, decode_text, extract_attributes_from_folder_modules, extract_json_from_folder
from pewtils.django import CacheHandler, reset_django_connection_wrapper, get_model, get_app_settings_folders

from django_learning.utils.dataset_extractors.document_coder_dataset import Extractor as DocumentCoderDatasetExtractor


class Extractor(DocumentCoderDatasetExtractor):

    def __init__(self, **kwargs):

        super(Extractor, self).__init__(**kwargs)

        self.coder_aggregation_function = kwargs.get("coder_aggregation_function", "mean")

        self.convert_to_discrete = kwargs.get("convert_to_discrete", False)
        self.threshold = kwargs.get("threshold", None)
        self.base_class_id = kwargs.get("base_class_id", None)
        if self.base_class_id and self.valid_label_ids:
            try: self.labels.get(pk=self.base_class_id)
            except:
                raise Exception("If you specify a base_class_id, it needs to belong to one of the selected questions")

        self.index_levels = ["document_id"]

    def _additional_steps(self, dataset, **kwargs):

        dataset = super(Extractor, self)._additional_steps(dataset, **kwargs)

        if self.coder_aggregation_function == "mean":
            dataset = dataset.groupby("document_id").mean().reset_index()
        else:
            raise Exception("Specify another aggregation function, fool!")
        for col in ["coder_is_mturk", "coder_id"]:
            if col in dataset.columns:
                del dataset[col]

        if self.convert_to_discrete:

            def get_max(x):

                max_col, max_val = sorted([(col, x[col]) for col in self.outcome_columns], key=lambda x: x[1], reverse=True)[0]
                if not self.threshold:
                    return max_col.split("_")[-1]
                elif max_val >= self.threshold:
                    return max_col.split("_")[-1]
                else:
                    return self.base_class_id

            dataset["label_id"] = dataset.apply(get_max, axis=1)
            for col in self.outcome_columns:
                del dataset[col]
            self.outcome_columns = []
            self.outcome_column = "label_id"
            self.discrete_classes = True

        return dataset