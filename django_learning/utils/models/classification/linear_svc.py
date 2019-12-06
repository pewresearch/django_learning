from sklearn.svm import LinearSVC
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html


def get_params():

    return {
        "model_class": LinearSVC(),
        "params": {
            'max_iter': (1000, ),
            'penalty': ('l2', ), # can also be 'l1'
            'class_weight' : (None, ), # can also be 'auto' or 'balanced'
            'loss': ('squared_hinge', ) # can also be 'hinge'
        }
    }