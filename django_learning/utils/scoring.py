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

from django_learning.utils import wmom
from pewanalytics.stats.irr import compute_scores
from pewtils import is_not_null


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
    """
    Given two compatible dataframes, computes scores that compare the two.

    :param dataset1: Dataset #1
    :param dataset2: Dataset #2
    :param document_column: Name of the column that contains document IDs
    :param outcome_column: Name of the column with the outcome variable
    :param weight_column: (Optional) name of the column with weights
    :param min_overlap: (default is 10) minimum number of documents that must be contained in each dataset
    :param discrete_classes: (default is True) if True, scores are calculated for each value separately; if False, \
        scores are calculated once across all values under the assumption that your outcome column is binary
    :param pos_label: (Optional) specify the value in your outcome variable that indicates a positive case
    :return: a dataframe of scores for each class
    """

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
    """
    Given a vertically-concatenated dataframe of codes from different coders, returns IRR scores

    :param dataset: A dataset with all of the codes across all coders
    :param document_column: Name of the column that contains document IDs
    :param outcome_column: Name of the column with the outcome variable
    :param coder_column: Name of the column that contains coder IDs
    :param weight_column: (Optional) name of the column with weights
    :param min_overlap: (default is 10) minimum number of documents that must be contained in each dataset
    :param discrete_classes: (default is True) if True, scores are calculated for each value separately; if False, \
        scores are calculated once across all values under the assumption that your outcome column is binary
    :param pos_label: (Optional) specify the value in your outcome variable that indicates a positive case
    :return: a dataframe of scores for each class
    """

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
    """
    Loops over the range (0, 1) and calculates IRR scores at different probability thresholds. Requires that your
    ``predicted_df`` has a ``probability`` column.
    :param predicted_df: a dataframe of predictions
    :param comparison_df: a dataframe to compare the predictions against (e.g. the true values)
    :param outcome_column: Name of the column with your outcome variable
    :param base_code: (default is None) the base/negative value in your outcome column
    :param pos_code: (default is None) the positive class in your outcome column
    :param weight_column: (Optional) name of the column with weights
    :param max_prob: (default is 1) maximum upper bound of the probability thresholds to test
    :param n_thresholds: (default is 20) number of thresholds to test across the range of (0, ``max_prob``)
    :return: A dataframe of scores with a ``threshold`` column indicating the threshold that was used
    """

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
    """
    Given a dataframe of scores with a ``threshold`` column (like that generated by ``get_probability_threshold_score_df``,
    although you could concatenate results from multiple datasets together), returns the optimal probability threshold.
    :param score_df: A dataframe of scores computed at different thresholds
    :param metric: (default is "precision_recall_min") name of the scoring function to use to determine the probability threshold
    :return: the probability threshold that maximizes the scoring metric
    """

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
    """
    Runs ``get_probability_threshold_score_df`` and ``get_probability_threshold_from_score_df`` in sequence.

    :param predicted_df: a dataframe of predictions
    :param comparison_df: a dataframe to compare the predictions against (e.g. the true values)
    :param outcome_column: Name of the column with your outcome variable
    :param base_code: (default is None) the base/negative value in your outcome column
    :param pos_code: (default is None) the positive class in your outcome column
    :param metric: (default is "precision_recall_min") name of the scoring function to use to determine the probability threshold
    :param weight_column: (Optional) name of the column with weights
    :param max_prob: (default is 1) maximum upper bound of the probability thresholds to test
    :param n_thresholds: (default is 20) number of thresholds to test across the range of (0, ``max_prob``)

    :return: optimal probability threshold
    """

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
    """
    Given a dataset of predictions with a ``probability`` column, applies a probability threshold and converts
    the probabilities into a discrete outcome column
    :param predicted_df: dataframe of predictions
    :param threshold: probability threshold between 0 and 1
    :param outcome_column: (default is "label_id") name of the outcome column
    :param base_code: (Optional) the base/negative code in your outcome column
    :param pos_code: the positive class in your outcome column
    :return: a dataframe with your outcome column converted into discrete values based on the threshold
    """

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

        if pos_code:
            predicted_df["pos_probability"] = predicted_df.apply(
                lambda x: x["probability"]
                if x[outcome_column] != base_code
                else 1.0 - x["probability"],
                axis=1,
            )
            predicted_df[outcome_column] = predicted_df.apply(
                lambda x: pos_code if x["pos_probability"] >= threshold else base_code,
                axis=1,
            )
            del predicted_df["pos_probability"]
        else:
            predicted_df[outcome_column] = base_code

        return predicted_df
