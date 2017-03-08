from sklearn.neighbors import KNeighborsClassifier
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


def get_params():

    return {
        "model_class": KNeighborsClassifier(),
        "params": {
            'weights': ('uniform', ), # can also be 'distance' or a custom callable
            'algorithm': ('auto', ), # can also be 'ball_tree', 'kd_tree', 'brute'
            'leaf_size': (30, )
        }
    }