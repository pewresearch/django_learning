from statsmodels.stats.inter_rater import cohens_kappa
from collections import defaultdict

def scorer(y_true, y_pred, sample_weight=None):

    result_dict = {0: defaultdict(int), 1: defaultdict(int)}
    for true, pred in zip(y_true, y_pred):
        result_dict[true][pred] += 1
    kappa = cohens_kappa(
        [
            [result_dict[0][0], result_dict[0][1]],
            [result_dict[1][0], result_dict[1][1]]
        ]
    )
    return kappa["kappa"]