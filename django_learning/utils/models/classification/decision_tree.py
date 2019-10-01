from __future__ import absolute_import

from sklearn.tree import DecisionTreeClassifier

# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html


def get_params():

    return {
        "model_class": DecisionTreeClassifier(),
        "params": {
            "criterion": ("gini",),  # can also be 'entropy'
            "splitter": ("best",),  # can also be 'random'
            "max_depth": (None,),
            "min_samples_split": (2,),
            "min_samples_leaf": (1,),
            "max_features": (None,),  # can also be 'auto', 'sqrt', 'log2'
        },
    }
