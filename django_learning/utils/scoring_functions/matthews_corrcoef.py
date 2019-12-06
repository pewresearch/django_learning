import pandas

from sklearn.metrics import matthews_corrcoef

def scorer(y_true, y_pred, sample_weight=None):

    try: smallest_code = y_true.value_counts(ascending=True).index[0]
    except: smallest_code = pandas.Series(y_true).value_counts(ascending=True).index[0]
    y_t = [1 if y == smallest_code else 0 for y in y_true]
    y_p = [1 if y == smallest_code else 0 for y in y_pred]

    return matthews_corrcoef(y_t, y_p, sample_weight=sample_weight)