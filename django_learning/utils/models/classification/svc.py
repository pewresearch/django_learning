from sklearn.svm import SVC
# http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html


def get_params():

    return {
        "model_class": SVC(),
        "params": {
            # 'kernel': ('linear', ),
            # 'max_iter': (1000, ),
            # 'penalty': ('l2', ), # can also be 'l1'
            # 'class_weight' : (None, ), # can also be 'auto' or 'balanced'
            # 'loss': ('squared_hinge', ) # can also be 'hinge'
        }
    }