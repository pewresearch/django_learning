import re, pandas, numpy

from django_learning.utils.regex_filters import regex_filters
from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):

    def __init__(self, *args, **kwargs):

        self.name = "regex_counts"

        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        self.regex_filter = regex_filters[self.params["regex_filter"]]()

        preprocessors = self.get_preprocessors()

        rows = []
        for index, row in X.iterrows():
            text = row['text']
            for p in preprocessors:
                text = p.run(text)
            matches = self.regex_filter.findall(text)
            count = float(len(matches))
            row = {
                "count": count,
                "has_match": 1 if count > 0 else 0,
                "count_sq": count*count,
                "count_log": numpy.log(count+1.0)
            }
            rows.append(row)


        df = pandas.DataFrame(rows)
        self.features = df.columns

        return df

    def fit(self, X, y=None, **fit_params):

        return self

    def get_feature_names(self):

        return self.features