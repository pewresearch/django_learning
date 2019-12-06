from statsmodels.stats.inter_rater import cohens_kappa
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score


def scorer(y_true, y_pred, sample_weight=None):

    return cohen_kappa_score(
        y_true,
        y_pred,
        sample_weight=sample_weight
    )