import pandas, time, numpy, copy

from tqdm import tqdm

from pewtils.django import get_model, reset_django_connection_wrapper
from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):

    def __init__(self, *args, **kwargs):

        self.name = "django_field_lookups"
        self.feature_names = []
        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        fields = ["pk"] + self.params["fields"]
        vals = Document.objects.filter(pk__in=X['pk'].values).values(*fields)
        df = pandas.DataFrame.from_records(vals)
        return df.merge(X, how="left", on="pk")

    def fit(self, X, y=None, **fit_params):

        self.feature_names = self.params["fields"]
        return self

    def get_feature_names(self):

        return self.feature_names

