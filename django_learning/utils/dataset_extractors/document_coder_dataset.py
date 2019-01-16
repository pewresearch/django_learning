import os, numpy, pandas

from tqdm import tqdm

from django_learning.utils.dataset_extractors.document_coder_label_dataset import Extractor as DocumentCoderLabelDatasetExtractor


class Extractor(DocumentCoderLabelDatasetExtractor):

    def __init__(self, **kwargs):

        super(Extractor, self).__init__(**kwargs)
        self.standardize_coders = kwargs.get("standardize_coders", False)
        self.index_levels = ["document_id", "coder_id"]

    def _additional_steps(self, dataset, **kwargs):

        dummies = pandas.get_dummies(dataset[self.outcome_column], prefix="label")
        self.outcome_columns = [c for c in dummies.columns if c.startswith("label_")]
        dataset = pandas.concat([dataset, dummies], axis=1)
        agg_dict = {l: sum for l in self.outcome_columns}
        agg_dict.update({
            "coder_name": lambda x: x.value_counts().index[0],
            "coder_is_mturk": lambda x: x.value_counts().index[0],
            "sampling_weight": lambda x: numpy.average(x),
            "date": lambda x: x.value_counts().index[0] if len(x.value_counts().index) > 0 else None
        })
        dataset = dataset.groupby(["document_id", "coder_id"]).agg(agg_dict).reset_index()  # this is in case a question allows multiple labels; need to consolidate
        self.outcome_column = None
        self.discrete_classes = False

        if self.standardize_coders:

            if len(dataset["coder_id"].unique()) > 1:

                def _standardize(group):
                    for col in self.outcome_columns:
                        mean = group[col].mean()
                        std = group[col].std(ddof=1)
                        group[col] = group[col].apply(lambda x: (x - mean) / std)
                    return group

                dataset = pandas.concat([_standardize(group) for _, group in
                                       tqdm(dataset.groupby("coder_id"), desc="Standardizing for coder bias")])

        return dataset
