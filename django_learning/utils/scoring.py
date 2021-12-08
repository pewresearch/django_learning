from __future__ import print_function
import itertools, numpy, pandas, copy

from collections import defaultdict
from nltk.metrics.agreement import AnnotationTask
from statsmodels.stats.inter_rater import cohens_kappa, fleiss_kappa
from sklearn.metrics import (
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    cohen_kappa_score,
)
from scipy.stats import ttest_ind

from pewanalytics.internal import wmom
from pewanalytics.stats.irr import compute_scores
from pewtils import is_not_null

#         scoring_function = None
#         if "scoring_function" in self.parameters["model"].keys():
#             scoring_function = self._get_scoring_function(
#                 self.parameters["model"]["scoring_function"],
#                 binary_base_code=smallest_code if len(y.unique()) == 2 else None
#             )
#     def _get_scoring_function(self, func_name, binary_base_code=None):
#
#         try:
#
#             from django_learning.utils.scoring_functions import scoring_functions
#             scoring_function = make_scorer(scoring_functions[func_name])
#
#         except:
#
#             if "regression" in str(self.__class__):
#                 func_map = {
#                     "mean_squared_error": (mean_squared_error, False, False),
#                     "r2": (r2_score, True, False)
#                 }
#                 func, direction, needs_proba = func_map[func_name]
#                 scoring_function = make_scorer(func, needs_proba=needs_proba, greater_is_better=direction)
#             elif binary_base_code:
#                 func_map = {
#                     "f1": (f1_score, True, False),
#                     "precision": (precision_score, True, False),
#                     "recall": (recall_score, True, False),
#                     "brier_loss": (brier_score_loss, False, True)
#                 }
#                 func, direction, needs_proba = func_map[func_name]
#                 scoring_function = make_scorer(func, needs_proba=needs_proba, greater_is_better=direction,
#                                                pos_label=binary_base_code)
#             else:
#                 if self.parameters["model"]["scoring_function"] == "f1":
#                     scoring_function = "f1_macro"
#                     # scoring_function = "f1_micro"
#                     # scoring_function = "f1_weighted"
#                 elif self.parameters["model"]["scoring_function"] == "precision":
#                     scoring_function = "precision"
#                 else:
#                     scoring_function = "recall"
#
#         return scoring_function


def compute_scores_from_datasets_as_coders(
    dataset1,
    dataset2,
    document_column,
    outcome_column,
    weight_column=None,
    min_overlap=10,
    discrete_classes=True,
    pos_label=None,
):

    dataset1 = copy.deepcopy(dataset1)
    dataset2 = copy.deepcopy(dataset2)

    dataset1["coder_id"] = "dataset1"
    dataset2["coder_id"] = "dataset2"

    if document_column == "index":
        dataset1["index"] = dataset1.index
        dataset2["index"] = dataset2.index

    dataset = pandas.concat([dataset1, dataset2], sort=False)

    return compute_scores_from_dataset(
        dataset,
        document_column,
        outcome_column,
        "coder_id",
        weight_column=weight_column,
        min_overlap=min_overlap,
        discrete_classes=discrete_classes,
        pos_label=pos_label,
    )


def compute_scores_from_dataset(
    dataset,
    document_column,
    outcome_column,
    coder_column,
    weight_column=None,
    min_overlap=10,
    discrete_classes=True,
    pos_label=None,
):

    dataset = copy.deepcopy(dataset)

    if not weight_column:
        weight_column = "_weight"
        dataset["_weight"] = 1.0

    for col in [document_column, outcome_column, coder_column, weight_column]:
        if col and col not in dataset.columns:
            raise Exception(
                "Column '{}' does not exist in the dataset provided".format(col)
            )

    index_levels = [document_column, coder_column]
    if len(dataset) != len(dataset.groupby(index_levels).count()):
        raise Exception("All {} combinations must be unique!".format(index_levels))

    outcome_values = numpy.unique(dataset[outcome_column])
    # if len(outcome_values) == 1:
    #     raise Exception("Your outcome column is constant (only one value)!")

    pos_labels = []
    if discrete_classes:
        pos_labels = outcome_values

    scores = []
    combo_count = 0
    if len(numpy.unique(dataset[coder_column])) == 2:
        itercoders = itertools.combinations(dataset[coder_column].unique(), 2)
    else:
        itercoders = itertools.permutations(dataset[coder_column].unique(), 2)
    for coder1, coder2 in itercoders:

        doc_ids = set(
            dataset[dataset[coder_column] == coder1][document_column].unique()
        ).intersection(
            set(dataset[dataset[coder_column] == coder2][document_column].unique())
        )
        code_subset = dataset[
            (dataset[coder_column].isin([coder1, coder2]))
            & (dataset[document_column].isin(doc_ids))
        ]

        if len(doc_ids) >= min_overlap:

            coder_df = code_subset[
                [coder_column, document_column, outcome_column, weight_column]
            ].reset_index()
            if not pos_label:
                scores.append(
                    compute_scores(
                        coder_df,
                        coder1,
                        coder2,
                        outcome_column,
                        document_column,
                        coder_column,
                        weight_column,
                        pos_label=None,
                    )
                )

            for pos in pos_labels:
                if not pos_label or pos == pos_label:
                    coder_df["{}__{}".format(outcome_column, pos)] = (
                        coder_df[outcome_column] == pos
                    ).astype(int)
                    scoreset = compute_scores(
                        coder_df,
                        coder1,
                        coder2,
                        "{}__{}".format(outcome_column, pos),
                        document_column,
                        coder_column,
                        weight_column,
                        pos_label=1,
                    )
                    scores.append(scoreset)

        combo_count += 1

    return pandas.DataFrame(scores)


def get_probability_threshold_score_df(
    predicted_df,
    comparison_df,
    outcome_column="label_id",
    base_code=None,
    pos_code=None,
    weight_column=None,
    max_prob=1,
    n_thresholds=20,
):

    predicted_df = copy.copy(predicted_df)
    threshold_scores = []
    for threshold in numpy.linspace(0, max_prob, n_thresholds, endpoint=False):
        temp_df = apply_probability_threshold(
            predicted_df,
            threshold,
            outcome_column=outcome_column,
            base_code=base_code,
            pos_code=pos_code,
        )
        scores = compute_scores_from_datasets_as_coders(
            comparison_df, temp_df, "index", outcome_column, weight_column=weight_column
        )
        scores["threshold"] = threshold
        threshold_scores.append(scores)

    score_df = pandas.concat(threshold_scores)

    return score_df


def get_probability_threshold_from_score_df(score_df, metric="precision_recall_min"):

    sorted_df = (
        score_df.groupby("threshold")
        .agg({metric: min})
        .sort_values(metric, ascending=False)
    )
    max_threshold = sorted_df[sorted_df[metric] == sorted_df[metric].max()].index.max()
    min_threshold = sorted_df[sorted_df[metric] == sorted_df[metric].max()].index.min()
    sorted_df = sorted_df[~sorted_df[metric].isnull()]

    return numpy.average(
        sorted_df[sorted_df[metric] == sorted_df[metric].max()].index.values
    )


def find_probability_threshold(
    predicted_df,
    comparison_df,
    outcome_column="label_id",
    base_code=None,
    pos_code=None,
    metric="precision_recall_min",
    weight_column=None,
    max_prob=1,
    n_thresholds=20,
):

    score_df = get_probability_threshold_score_df(
        predicted_df,
        comparison_df,
        outcome_column=outcome_column,
        base_code=base_code,
        pos_code=pos_code,
        weight_column=weight_column,
        max_prob=max_prob,
        n_thresholds=n_thresholds,
    )
    return get_probability_threshold_from_score_df(score_df, metric=metric)


def apply_probability_threshold(
    predicted_df, threshold, outcome_column="label_id", base_code=None, pos_code=None
):

    if (
        "probability" not in predicted_df.columns
        or predicted_df["probability"].isnull().astype(int).sum() > 0
    ):

        print(
            "This model doesn't appear to produce probabilities or some are missing; cannot apply a threshold"
        )
        print("apply_probability_threshold will be skipped")

    else:

        predicted_df = copy.copy(predicted_df)
        if base_code == None:
            base_code = (
                predicted_df[outcome_column]
                .value_counts()
                .sort_values(ascending=False)
                .index[0]
            )
            print(
                "apply_probability_threshold: 'base_code' not provided, setting base_code to most frequent code"
            )
        if not pos_code:
            print(
                "Probability thresholding currently only works with binary classifications; setting all predictions to base_code"
            )
            print("Pass a pos_code if you wish to override it")

        predicted_df["probability"] = predicted_df.apply(
            lambda x: x["probability"]
            if x[outcome_column] != base_code
            else 1.0 - x["probability"],
            axis=1,
        )
        if pos_code:
            predicted_df[outcome_column] = predicted_df.apply(
                lambda x: pos_code if x["probability"] >= threshold else base_code,
                axis=1,
            )
        else:
            predicted_df[outcome_column] = base_code

        return predicted_df
