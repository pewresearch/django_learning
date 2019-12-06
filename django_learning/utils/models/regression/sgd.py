from sklearn.linear_model import SGDRegressor


def get_params():

    return {
        "model_class": SGDRegressor(),
        "params": {
            # "loss": ('squared_loss', )
        }
    }