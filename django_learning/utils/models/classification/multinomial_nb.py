from __future__ import absolute_import

from sklearn.naive_bayes import MultinomialNB


def get_params():

    return {"model_class": MultinomialNB(), "params": {}}
