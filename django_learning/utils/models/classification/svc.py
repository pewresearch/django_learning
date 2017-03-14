from sklearn.svm import SVC

# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html


def get_params():

    return {
        "model_class": SVC(),
        "params": {
            "kernel": ("linear", "rbf"),
            "class_weight": ("balanced",),  # can be None, 'auto', or 'balanced'
            "gamma": ("auto",),
        },
    }
