from __future__ import print_function
import pandas, time, numpy, copy

from tqdm import tqdm

from django_pewtils import get_model, reset_django_connection_wrapper
from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):
    def __init__(self, *args, **kwargs):

        self.name = "word2vec"
        self.models = None

        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        X = copy.deepcopy(X)
        for p in self.get_preprocessors():
            X["text"] = X["text"].apply(p.run)

        # return pandas.concat([m.apply_model_to_dataframe(X) for m in self.models.values()], axis=1)
        # TODO: split X up by the document_type column, and send the subsets to their appropriate models
        raise Exception

    def fit(self, X, y=None, **fit_params):

        if not self.models and self.params["document_types"]:
            self.models = {}
            for doc_type in self.params["document_types"]:
                print("Loading Word2Vec {} model".format(doc_type))
                try:
                    self.models[doc_type] = get_model("Word2VecModel").objects.get(
                        document_type=doc_type,
                        use_skipgrams=self.params.get("use_skipgrams", True),
                        use_sentences=self.params.get("use_sentences", False),
                        politicians_only=self.params.get("politicians_only", True),
                    )
                except:
                    print("Couldn't load {} Word2Vec model!".format(doc_type))
                    raise Exception

        return self

    def get_feature_names(self):

        feature_names = []
        for doc_type, model in self.models.iteritems():
            feature_names.extend(model.get_feature_names())
        if self.params["feature_name_prefix"]:
            return [
                "{}_{}".format(self.params["feature_name_prefix"], f)
                for f in feature_names
            ]
        else:
            return feature_names
