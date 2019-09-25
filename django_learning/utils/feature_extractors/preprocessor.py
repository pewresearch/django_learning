import pandas, time, numpy, copy

from tqdm import tqdm

from django_pewtils import get_model, reset_django_connection_wrapper
from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):
    def __init__(self, *args, **kwargs):

        self.name = "preprocessor"
        self.feature_names = []
        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        X = copy.deepcopy(X)
        for p in self.get_preprocessors():
            X["text"] = X["text"].apply(p.run)

        return X

    def fit(self, X, y=None, **fit_params):

        self.feature_names = X.columns
        return self

    def get_feature_names(self):

        return self.feature_names
