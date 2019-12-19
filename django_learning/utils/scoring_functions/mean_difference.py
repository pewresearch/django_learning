import numpy


def scorer(y_true, y_pred, sample_weight=None):

    avgs = [
        numpy.average([int(y) for y in y_true], weights=sample_weight),
        numpy.average([int(y) for y in y_pred], weights=sample_weight),
    ]
    lower = min(avgs)
    higher = max(avgs)
    return lower / higher
