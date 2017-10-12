import pandas
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer

from pewtils.stats import wmom
from pewtils.django import get_model

from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):

    def __init__(self, *args, **kwargs):

        self.name = "tfidf"
        self.vectorizer = TfidfVectorizer()
        if kwargs.get("normalize_document_types", False):
            self.normalize_document_types = True
        else:
            self.normalize_document_types = False

        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        # print "Preprocessing text (transform)"
        # if "text" not in X.columns and "document_id" in X.columns:
        #     X['text'] = X['document_id'].map(lambda x: get_model("Document", app_name="django_learning").objects.get(pk=x).text)
        text = X['text']
        for p in self.get_preprocessors():
            text = text.apply(p.run)

        # print "Transforming text to TF-IDF matrix (transform)"
        ngrams = self.vectorizer.transform(text, **transform_params)

        if hasattr(self, "normalize_document_types") and self.normalize_document_types:

            # print "Normalizing across document types (transform)"
            ngrams = pandas.DataFrame(ngrams.todense(), index=X.index)
            for doctype, group in X.groupby("document_type"):
                for col in ngrams.columns:
                    ngrams.ix[group.index, col] = ((ngrams[col] - self.mean_mapper[doctype][col]) / self.std_mapper[doctype][col])

        return ngrams

    def fit(self, X, y=None, **fit_params):

        # print "Preprocessing text (fit)"
        text = X['text']
        for p in self.get_preprocessors():
            text = text.apply(p.run)

        # print "Fitting TF-IDF matrix (fit)"
        self.vectorizer.fit(text, y, **fit_params)

        if hasattr(self, "normalize_document_types") and self.normalize_document_types:

            # print "Computing normalization parameters (fit)"
            ngrams = pandas.DataFrame(self.vectorizer.transform(text).todense(), index=X.index)
            self.mean_mapper = defaultdict(dict)
            self.std_mapper = defaultdict(dict)
            for doctype, group in X.groupby("document_type"):
                for col in ngrams.columns:
                    mean, err, std = wmom(ngrams[col][group.index], group["sampling_weight"], calcerr=True, sdev=True)
                    self.mean_mapper[doctype][col] = mean
                    self.std_mapper[doctype][col] = std
            # self.mean_mapper[doctype] = ngrams[group.index].mean(axis=1).to_dict()
            # self.std_mapper[doctype] = ngrams[group.index].std(axis=1).to_dict()

        return self

    def get_feature_names(self):

        if self.params["feature_name_prefix"]:
            return ["%s__%s" % (self.params["feature_name_prefix"], ngram) for ngram in self.vectorizer.get_feature_names()]
        else:
            return self.vectorizer.get_feature_names()

    def get_params(self, *args, **kwargs):

        params = super(Extractor, self).get_params(**kwargs)
        params.update(self.vectorizer.get_params(*args, **kwargs))

        return params

    def set_params(self, *args, **kwargs):

        vec_params = self.vectorizer.get_params()
        for k in kwargs:
            if k in vec_params:
                vec_params[k] = kwargs[k]
        # del vec_params["preprocessor"]
        self.vectorizer.set_params(**vec_params)

        super(Extractor, self).set_params(*args, **kwargs)

        return self

