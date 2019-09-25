from __future__ import absolute_import

try:

    from xgboost.sklearn import XGBClassifier

    def get_params():

        return {
            "model_class": XGBClassifier(),
            "params": {
                "max_depth": [6],  # default 3
                "n_estimators": [150],  # default 100
                # 'kernel': ('linear', ),
                # 'max_iter': (1000, ),
                # 'penalty': ('l2', ), # can also be 'l1'
                # 'class_weight' : (None, ), # can also be 'auto' or 'balanced'
                # 'loss': ('squared_hinge', ) # can also be 'hinge'
            },
        }


except ImportError:
    pass
