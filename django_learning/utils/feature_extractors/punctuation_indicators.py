from __future__ import absolute_import

import re, pandas

from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):
    def __init__(self, *args, **kwargs):

        self.name = "punctuation_indicators"

        self.regexes = [
            ("dollars", r"\$([0-9]{1,3}(?:(?:\,[0-9]{3})+)?(?:\.[0-9]{1,2})?)\s"),
            ("dollars_alt", r"\$[0-9]{1,3}((\,[0-9]{3})+)?\s"),
            ("amounts", r"thousand*|million*|billion*|hundred*"),
            ("exclamation", r"\!"),
        ]

        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        text = X["text"]
        for p in self.get_preprocessors():
            text = text.apply(p.run)

        regexes = [(n, re.compile(r, re.IGNORECASE)) for n, r in self.regexes]
        rows = []
        for text in text:
            row = []
            for name, regex in regexes:
                matches = [m for m in regex.findall(text) if m != ""]
                row.extend([len(matches), 1 if len(matches) > 0 else 0])
            rows.append(row)

        return pandas.DataFrame(rows, columns=self.get_feature_names())

    def fit(self, X, y=None, **fit_params):

        return self

    def get_feature_names(self):

        names = []
        for name, regex in self.regexes:
            names.extend(["{}_count".format(name), "{}_any".format(name)])

        if self.params["feature_name_prefix"]:
            return ["%s__%s" % (self.params["feature_name_prefix"], n) for n in names]
        else:
            return names
