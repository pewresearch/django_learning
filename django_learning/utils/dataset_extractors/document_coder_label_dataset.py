import os, numpy, pandas, inspect, copy

from tqdm import tqdm

from django.conf import settings

from pewtils import classproperty, is_not_null, is_null, decode_text, extract_attributes_from_folder_modules, extract_json_from_folder
from pewtils.django import CacheHandler, reset_django_connection_wrapper, get_model, get_app_settings_folders
from pewtils.sampling import compute_balanced_sample_weights

from django_learning.utils.dataset_code_filters import dataset_code_filters
from django_learning.utils.dataset_document_filters import dataset_document_filters
from django_learning.utils.dataset_coder_filters import dataset_coder_filters
from django_learning.utils.balancing_variables import balancing_variables
from django_learning.utils.dataset_extractors import DatasetExtractor


class Extractor(DatasetExtractor):

    def __init__(self,
                 project_name=None,
                 sample_names=None,
                 question_names=None,
                 code_filters=None,
                 coder_filters=None,
                 document_filters=None,
                 balancing_variables=None,
                 **kwargs
                 ):

        super(Extractor, self).__init__(**kwargs)

        self.project = get_model("Project", app_name="django_learning").objects.get(name=project_name)
        self.samples = self.project.samples.filter(name__in=sample_names)
        self.questions = self.project.questions.filter(name__in=question_names)
        self.labels = get_model("Label", app_name="django_learning").objects.filter(question__in=self.questions.all())

        self.code_filters = code_filters if code_filters else []
        self.coder_filters = coder_filters if coder_filters else []
        self.document_filters = document_filters if document_filters else []

        self.balancing_variables = balancing_variables if balancing_variables else []

        self.raw_codes = get_model("Code", app_name="django_learning").objects \
            .filter(sample_unit__sample__in=self.samples.all())\
            .filter(label__in=self.labels.all())

        self.index_levels = ["document_id", "coder_id", "label_id"]
        self.outcome_column = None
        self.outcome_columns = None
        self.discrete_classes = None
        self.valid_label_ids = True  # if this is still true after processing everything, and discrete_classes=True, then you can use it for document classifications
        # although, maybe the thing to do is to also do validation when you pass a dataset to the model
        # and just have it confirm that the outcome column consists of label primary keys for a single question
        # and that all of the documents belong to its own frame

        frame_ids = set(list(self.samples.values_list("frame_id", flat=True)))
        if len(frame_ids) == 1:
            self.sampling_frame = get_model("SamplingFrame", app_name="django_learning").objects.get(pk__in=frame_ids)
        else:
            raise Exception("All of your samples must be belong to the same sampling frame")

    def set_outcome_column(self, outcome_col):

        if outcome_col == self.outcome_column: pass
        elif self.outcome_columns and outcome_col in self.outcome_columns:
            self.outcome_column = outcome_col
        else:
            raise Exception("'{}' is not a valid outcome column for this dataset".format(outcome_col))

    def get_hash(self, **kwargs):

        hash_key = super(Extractor, self).get_hash(**kwargs)
        hash_key += str(inspect.getsourcelines(self._additional_steps)) + str(self.samples) + str(self.questions) + str(self.code_filters)

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

        dataset = self._get_raw_codes()

        if len(dataset) > 0:

            dataset = self._apply_filters(dataset)

            if self.questions.count() > 1:

                print "Multiple questions provided, concatenating labels into single string representation"
                dummies = pandas.get_dummies(dataset[self.outcome_column], prefix="label")
                label_cols = [c for c in dummies.columns if c.startswith("label_")]
                dataset = pandas.concat([dataset, dummies], axis=1)
                agg_dict = {l: sum for l in label_cols}
                agg_dict.update({
                    "coder_name": lambda x: x.value_counts().index[0],
                    "coder_is_mturk": lambda x: x.value_counts().index[0],
                    "sampling_weight": lambda x: numpy.average(x),
                    "date": lambda x: x.value_counts().index[0]
                })
                grouped = dataset.groupby(["document_id", "coder_id"]).agg(agg_dict)
                grouped['label_id'] = grouped.apply(lambda x: "".join(["1" if x[l] > 0 else "0" for l in label_cols]), axis=1)
                dataset = grouped.reset_index()
                for col in label_cols:
                    del dataset[col]
                self.valid_label_ids = False

            dataset = self._additional_steps(dataset, **kwargs)
            self._test_index(dataset)
            dataset = self._add_document_data(dataset)
            self._add_balancing_weights(dataset)

            if self.outcome_column:
                print "Extracted dataset: outcome_column '{}', discrete='{}', valid_label_ids='{}'".format(
                    self.outcome_column, self.discrete_classes, self.valid_label_ids
                )
            else:
                print "Extracted dataset: outcome_columns '{}', discrete='{}', valid_label_ids='{}'".format(
                    self.outcome_columns, self.discrete_classes, self.valid_label_ids
                )

        return dataset

    def _additional_steps(self, dataset, **kwargs):

        return dataset

    def _get_raw_codes(self):

        columns = [
            "pk",
            "coder_id",
            "coder__name",
            "coder__is_mturk",
            "document_id",
            "document__date",
            # "sample_unit_id",
            "sample_unit__weight",
            "label_id",
            "label__value",
            "label__question_id",
            "label__question__name"
        ]
        dataset = pandas.DataFrame.from_records(
            self.raw_codes.values(*columns)
        )
        dataset = dataset.rename(columns={
            "coder__name": "coder_name",
            "coder__is_mturk": "coder_is_mturk",
            "document__date": "date",
            "sample_unit__weight": "sampling_weight",
            "label__value": "label_value",
            "label__question_id": "question_id",
            "label__question_name": "question_name"
        })

        self.outcome_column = "label_id"
        self.discrete_classes = True

        return dataset

    def _apply_filters(self, dataset):


        for filter_name, filter_args, filter_kwargs in self.code_filters:
            dataset = dataset_code_filters[filter_name](self, dataset, *filter_args, **filter_kwargs)
        for filter_name, filter_args, filter_kwargs in self.coder_filters:
            dataset = dataset_coder_filters[filter_name](self, dataset, *filter_args, **filter_kwargs)
        for filter_name, filter_args, filter_kwargs in self.document_filters:
            dataset = dataset_document_filters[filter_name](self, dataset, *filter_args, **filter_kwargs)

        return dataset

    def _add_document_data(self, dataset):

        doc_columns = [
            "document_id",
            "document__text"
        ]
        if "date" not in dataset.columns:
            doc_columns.append("document__date")
        doc_types = get_model("Document", app_name="django_learning").objects.document_types()
        for doc_type in doc_types:
            doc_columns.append("sample_unit__document__{}__pk".format(doc_type))
        docs = pandas.DataFrame.from_records(
            self.raw_codes.values(*doc_columns).distinct()
        )
        docs = docs.rename(columns={"document__text": "text"})
        if "document__date" in docs.columns:
            docs = docs.rename(columns={"document__date": "date"})
        for doc_type in doc_types:
            docs.ix[~docs["sample_unit__document__{}__pk".format(doc_type)].isnull(), "document_type"] = doc_type
            del docs["sample_unit__document__{}__pk".format(doc_type)]
        dataset = dataset.merge(docs, how="left", on="document_id")
        self.document_types = list(dataset['document_type'].unique())

        return dataset

    def _add_balancing_weights(self, dataset):

        sample = copy.copy(dataset)

        weight_var_names = []
        for mapper_name in self.balancing_variables:
            # balancing_module = importlib.import_module(
            #     "logos.learning.utils.balancing_variables.{0}".format(mapper_name))
            sample[mapper_name] = sample.apply(balancing_variables[mapper_name], axis=1)
            # sample[mapper_name] = sample.apply(balancing_module.var_mapper, axis=1)
            weight_var_names.append(mapper_name)

        if len(weight_var_names) > 0:

            print "Computing balanced weights across combined variable strata: {}".format(weight_var_names)

            weight_vars = []
            for var in weight_var_names:
                var_sample = sample.dropna(subset=[var])[[var]]
                dummies = pandas.get_dummies(var_sample, prefix=var, columns=[var])
                weight_vars.extend([d for d in dummies.columns if d.startswith(var)])
                # sample = sample.merge(dummies, how="left")
                sample = sample.join(dummies, how="left")

            dataset['balancing_weight'] = compute_balanced_sample_weights(sample, weight_vars).fillna(1.0) #, weight_column="sampling_weight") # DocumentClassificationModel should do this now

        return dataset