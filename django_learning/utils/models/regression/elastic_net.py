from sklearn.linear_model import ElasticNet


def get_params():

    return {
        "model_class": ElasticNet(),
        "params": {
            # "normalize": (False, )
        }
    }