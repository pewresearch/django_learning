import pandas, time, numpy, copy

from tqdm import tqdm

from django_pewtils import get_model, reset_django_connection_wrapper
from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):

    def __init__(self, *args, **kwargs):

        self.name = "django_field_lookups"
        self.feature_names = []
        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        fields = ["pk"] + self.params["fields"]
        vals = get_model("Document").objects.filter(pk__in=X['document_id'].values).values(*fields)
        df = pandas.DataFrame.from_records(vals)
        for col in df.columns:
            if col != "document_id":
                df[col] = df[col].astype(float)
        return X[['document_id']].merge(df, how="left", left_on="document_id", right_on="pk")[[f for f in fields if f != "pk"]]

    def fit(self, X, y=None, **fit_params):

        self.feature_names = self.params["fields"]
        return self

    def get_feature_names(self):

        return self.feature_names

