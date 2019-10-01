from __future__ import absolute_import

from sklearn.linear_model import LinearRegression


def get_params():

    return {"model_class": LinearRegression(), "params": {"normalize": (False,)}}
