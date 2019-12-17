from __future__ import print_function
import datetime
import pandas as pd

from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.metrics import precision_recall_fscore_support

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import *


class ModelsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()
        set_up_test_sample("test_sample", 250)

    def test_loading(self):

        from django_learning.utils.models import models

        # NOTE: in Python 2, sklearn used relative importing in the ensemble module
        # So the decision tree, gradient boosting and random forest models can't import

        for val in [
            "classification_decision_tree",  # doesn't work in Python 2
            "classification_gradient_boosting",  # doesn't work in Python 2
            "classification_k_neighbors",
            "classification_linear_svc",
            "classification_multinomial_nb",
            "classification_random_forest",  # doesn't work in Python 2
            "classification_sgd",
            "classification_svc",
            "classification_xgboost",
        ]:
            self.assertIn(val, models.keys())
            self.assertIsNotNone(models[val]())

    def test_models(self):

        from django_learning.utils.models import models
        from django_learning.utils.feature_extractors import feature_extractors
        from pewanalytics.text import TextCleaner

        df = extract_dataset("document_dataset", params={"threshold": .1})
        c = TextCleaner()
        df['text'] = df['text'].map(c.clean)
        tfidf = feature_extractors["tfidf"](max_df=.9, min_df=10, max_features=None).fit_transform(df)
        base_class_id = (
            get_model("Question", app_name="django_learning")
                .objects.filter(project__name="test_project")
                .get(name="test_checkbox")
                .labels.get(value="0")
                .pk
        )
        # df['outcome'] = (df['label_id']!=str(base_class_id)).astype(int)
        df['outcome'] = df['text'].str.contains(r"film", flags=re.IGNORECASE).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(
            tfidf,
            df['outcome'],
            test_size = 0.25,
            random_state = 42,
            stratify=df['outcome'].values
        )
        for val in [
            "classification_decision_tree",  # doesn't work in Python 2
            "classification_gradient_boosting",  # doesn't work in Python 2
            "classification_k_neighbors",
            "classification_linear_svc",
            "classification_multinomial_nb",
            "classification_random_forest",  # doesn't work in Python 2
            "classification_sgd",
            "classification_svc",
            "classification_xgboost",
        ]:
            params = models[val]()
            model = params['model_class']
            for params in ParameterGrid(params["params"]):
                model.set_params(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics  = precision_recall_fscore_support(y_test, y_pred)
                for scores in metrics:
                    for score in scores:
                        self.assertGreater(score, 0.0)


    def tearDown(self):

        from django.conf import settings
        import shutil

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
