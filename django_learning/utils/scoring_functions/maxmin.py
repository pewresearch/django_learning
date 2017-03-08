from sklearn.metrics import precision_score, recall_score

def scorer(y_true, y_pred, sample_weight=None):

    metrics = []
    for value in set(y_true.values):
        metrics.append(precision_score(y_true, y_pred, average='binary', pos_label=value, sample_weight=sample_weight))
        metrics.append(recall_score(y_true, y_pred, average='binary', pos_label=value, sample_weight=sample_weight))
    return min(metrics)