from sklearn.linear_model import SGDClassifier

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html


def get_params():

    return {
        "model_class": SGDClassifier(),
        "params": {
            "loss": (
                "hinge",
            ),  # can also be 'log', 'modified_huber', 'squared_hinge', 'perceptron'
            "penalty": ("l2",),  # can also be 'none', 'l1', 'elasticnet'
            "l1_ratio": (0.15,),
            "alpha": (0.0001,),
            "class_weight": ("balanced",),  # can also be 'auto' or 'balanced'
            "average": (False,),
        },
    }
