from __future__ import absolute_import

from sklearn.ensemble import GradientBoostingClassifier

# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html


def get_params():

    return {
        "model_class": GradientBoostingClassifier(),
        "params": {
            "loss": ("deviance",),  # can also be 'exponential'
            "learning_rate": (0.1,),
            "n_estimators": (100,),
            "max_depth": (None,),
            "subsample": (1.0,),
            "min_samples_split": (2,),
            "min_samples_leaf": (1,),
            "max_features": (None,),
        },
    }
