import itertools, numpy, pandas

from collections import defaultdict
from nltk.metrics.agreement import AnnotationTask
from statsmodels.stats.inter_rater import cohens_kappa
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
from scipy.stats import ttest_ind

from pewtils.stats import wmom

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


def compute_scores_from_datasets_as_coders(dataset1, dataset2, document_column, outcome_column, weight_column=None, min_overlap=10):

    dataset1["coder_id"] = "dataset1"
    dataset2["coder_id"] = "dataset2"

    dataset = pandas.concat([dataset1, dataset2])

    return compute_scores_from_dataset(dataset, document_column, outcome_column, "coder_id", weight_column=weight_column, min_overlap=min_overlap)


def compute_scores_from_dataset(dataset, document_column, outcome_column, coder_column, weight_column=None, min_overlap=10):

    if not weight_column:
        weight_column = "_weight"
        dataset["_weight"] = 1.0

    for col in [document_column, outcome_column, coder_column, weight_column]:
        if col and col not in dataset.columns:
            raise Exception("Column '{}' does not exist in the dataset provided".format(col))

    index_levels = [document_column, coder_column]
    if len(dataset) != len(dataset.groupby(index_levels).count()):
        raise Exception("All {} combinations must be unique!".format(index_levels))

    outcome_values = numpy.unique(dataset[outcome_column])
    pos_label, base_label = None, None
    if len(outcome_values) > 2:
        binary = False
    elif len(outcome_values) == 2:
        binary = True
        try: already_binary = [int(v) for v in sorted(list(set(dataset[outcome_column].values)))] == [0, 1]
        except ValueError: already_binary = False
        pos_label = 1
        if not already_binary:
            pos = dataset[outcome_column].value_counts().index[0]
            neg = dataset[outcome_column].value_counts().index[-1]
            dataset[outcome_column] = (dataset[outcome_column] == pos).astype(int)
        else:
            dataset[outcome_column] = dataset[outcome_column].astype(int)
    else:
        raise Exception("Your outcome column is constant (only one value)!")

    scores = []
    combo_count = 0
    for coder1, coder2 in itertools.permutations(dataset[coder_column].unique(), 2):

        doc_ids = dataset[dataset[coder_column] == coder1][document_column].unique()
        code_subset = dataset[(dataset[coder_column].isin([coder1, coder2])) & (dataset[document_column].isin(doc_ids))]

        if len(code_subset) >= min_overlap:

            coder_df = code_subset[[coder_column, document_column, outcome_column, weight_column]]
            coder1_df = coder_df[coder_df[coder_column] == coder1]
            coder1_df.index = coder1_df[document_column]
            coder2_df = coder_df[coder_df[coder_column] == coder2]
            coder2_df.index = coder2_df[document_column]
            coder2_df = coder2_df.ix[coder1_df.index]

            alpha = AnnotationTask(data=coder_df[[coder_column, document_column, outcome_column]].as_matrix())
            try:
                alpha = alpha.alpha()
            except ZeroDivisionError:
                alpha = None

            row = {
                "coder1": coder1,
                "coder2": coder2,
                "alpha": alpha
            }

            try: row["accuracy"] = accuracy_score(coder1_df[outcome_column], coder2_df[outcome_column], sample_weight=coder1_df[weight_column]),
            except ValueError: row["accuracy"] = None

            try: row["f1"] = f1_score(coder1_df[outcome_column], coder2_df[outcome_column], pos_label=pos_label, sample_weight=coder1_df[weight_column])
            except ValueError: row["f1"] = None

            try: row["precision"] = precision_score(coder1_df[outcome_column], coder2_df[outcome_column], pos_label=pos_label, sample_weight=coder1_df[weight_column])
            except ValueError: row["precision"] = None

            try: row["recall"] = recall_score(coder1_df[outcome_column], coder2_df[outcome_column], pos_label=pos_label, sample_weight=coder1_df[weight_column]),
            except ValueError: row["recall"] = None

            try: row["matthews_corrcoef"] = matthews_corrcoef(coder1_df[outcome_column], coder2_df[outcome_column], sample_weight=coder1_df[weight_column])
            except ValueError: row["matthews_corrcoef"] = None

            try: row["roc_auc"] = roc_auc_score(coder1_df[outcome_column], coder2_df[outcome_column], sample_weight=coder1_df[weight_column]) if len(numpy.unique(coder1_df[outcome_column])) > 1 and len(numpy.unique(coder2_df[outcome_column])) > 1 else None
            except  ValueError: row["roc_auc"] = None

            for labelsetname, labelset in [
                ("coder1", coder1_df[outcome_column]),
                ("coder2", coder2_df[outcome_column])
            ]:
                # unweighted = wmom(codeset, [1.0 for x in codeset], calcerr=True, sdev=True)
                try: weighted = wmom(labelset, coder1_df[weight_column], calcerr=True, sdev=True)
                except TypeError: weighted = wmom(labelset.astype(int), coder1_df[weight_column], calcerr=True, sdev=True)
                # for valname, val in zip(["mean", "err", "std"], list(unweighted)):
                #     row["{}_{}".format(codesetname, valname)] = val
                for valname, val in zip(["mean", "err", "std"], list(weighted)):
                    row["{}_{}".format(labelsetname, valname)] = val

            try: row["ttest_t"], row["ttest_p"] = ttest_ind(coder1_df[outcome_column], coder2_df[outcome_column])
            except TypeError: row["ttest_t"], row["ttest_p"] = ttest_ind(coder1_df[outcome_column].astype(int), coder2_df[outcome_column].astype(int))
            if row["ttest_p"] > .05:
                row["ttest_pass"] = 1
            else:
                row["ttest_pass"] = 0

            row["pct_agree"] = numpy.average([1 if c[0] == c[1] else 0 for c in zip(coder1_df[outcome_column], coder2_df[outcome_column])])

            if binary:
                if sum(coder1_df[outcome_column]) > 0 and sum(coder2_df[outcome_column]) > 0:
                    result_dict = {0: defaultdict(int), 1: defaultdict(int)}
                    for pred, true in zip(coder1_df[outcome_column], coder2_df[outcome_column],):
                        result_dict[pred][true] += 1
                    kappa = cohens_kappa([
                        [result_dict[0][0], result_dict[0][1]],
                        [result_dict[1][0], result_dict[1][1]]
                    ])
                    row["kappa"] = kappa["kappa"]
                    row["kappa_err"] = kappa["std_kappa"]
            else:
                row["kappa"] = None
                row["kappa_err"] = None

            scores.append(row)

        combo_count += 1

    return pandas.DataFrame(scores)