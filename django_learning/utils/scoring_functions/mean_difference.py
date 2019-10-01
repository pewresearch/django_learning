from __future__ import absolute_import

import numpy


def scorer(y_true, y_pred, sample_weight=None):

    avgs = [
        numpy.average([int(y) for y in y_true], weights=sample_weight),
        numpy.average([int(y) for y in y_pred], weights=sample_weight),
    ]
    lower = min(avgs)
    higher = max(avgs)
    return lower / higher
    # return 1.0 / numpy.average([int(y) for y in y_true], weights=sample_weight) - numpy.average([int(y) for y in y_pred], weights=sample_weight)
    # print "True avg: {}, pred avg: {}".format(numpy.average([int(y) for y in y_true], weights=sample_weight), numpy.average([int(y) for y in y_pred], weights=sample_weight))
    # return 1.0 / abs(numpy.average([int(y) for y in y_true], weights=sample_weight) / numpy.average([int(y) for y in y_pred], weights=sample_weight))
