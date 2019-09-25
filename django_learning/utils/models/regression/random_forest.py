from sklearn.ensemble import RandomForestRegressor


def get_params():

    return {"model_class": RandomForestRegressor(), "params": {"n_estimators": (10,)}}
