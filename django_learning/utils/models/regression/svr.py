from sklearn.svm import SVR


def get_params():

    return {
        "model_class": SVR(),
        "params": {
            "kernel": ('linear', ) # linear, poly, rbf, signmoid
        }
    }