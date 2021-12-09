Models
======

Model utilities in Django Learning are simple wrappers around sklearn classification algorithms, combined with
hyperparameters you want to grid search. For example, the basic implimentation of LinearSVC in Django Learning looks
like this:

.. code:: python

    from sklearn.svm import LinearSVC

    def get_params():

        return {
            "model_class": LinearSVC(),
            "params": {
                "max_iter": (1000,),
                "penalty": ("l2",),  # can also be 'l1'
                "class_weight": (None,),  # can also be 'auto' or 'balanced'
                "loss": ("squared_hinge",),  # can also be 'hinge'
            },
        }

You can create your own models using any classification algorithm that's compatible with sklearn, and add whatever
hyperparameter options you want to grid search. You'll see above that we don't test different hyperparameters by
default, for the sake of efficiency. But you can easily create your own models file and put it in one of your
``settings.DJANGO_LEARNING_MODELS`` folders, and then refer to it by name when building out your pipeline.

Built-in models
----------------

Right now, Django Learning is only configured for classification models. However, we could add regressions in the
future, so the built-in models are namespaced with the ``classification_`` prefix:

    * ``classification_decision_tree``: wrapper around sklearn's ``DecisionTreeClassifier``
    * ``classification_gradient_boosting``: wrapper around sklearn's ``GradientBoostingClassifier``
    * ``classification_k_neighbors``: wrapper around sklearn's ``KNeighborsClassifier``
    * ``classification_linear_svc``: wrapper around sklearn's ``LinearSVC``
    * ``classification_multinomial_nb``: wrapper around sklearn's ``MultinomialNB``
    * ``classification_random_forest``: wrapper around sklearn's ``RandomForestClassifier``
    * ``classification_sgd``: wrapper around sklearn's ``SGDClassifier``
    * ``classification_svc``: wrapper around sklearn's ``SVC``
    * ``classification_xgboost``: wrapper around xgboost's sklearn-compatible ``XGBClassifier``
