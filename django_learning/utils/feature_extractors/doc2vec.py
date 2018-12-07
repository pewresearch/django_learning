import pandas, time

from tqdm import tqdm

from django_pewtils import get_model, reset_django_connection_wrapper
from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):

    def __init__(self, *args, **kwargs):

        self.name = "doc2vec"
        self.models = None

        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        preprocessors = self.get_preprocessors()

        vecs = []
        for index, row in tqdm(X.iterrows(), desc="Computing Doc2Vec features"):
            new_row = self.get_row_cache(str(row['pk']))
            if not new_row:
                text = row['text']
                for p in preprocessors:
                    text = p.run(text)
                new_row = []
                for doc_type, model in self.models.iteritems():
                    new_row.extend(model.infer_vector(text.split()))
                self.set_row_cache(str(row['pk']), new_row)
            if type(new_row) != list:
                print "WAHH!"
                print new_row
            vecs.append(new_row)

        return pandas.DataFrame(vecs, columns=self.get_feature_names())

    def fit(self, X, y=None, **fit_params):

        if not self.models and self.params["document_types"]:
            self.models = {}
            for doc_type in self.params["document_types"]:
                print "Loading Doc2Vec {} model".format(doc_type)
                self.models[doc_type] = self.get_row_cache("model")
                if self.models[doc_type] == None:
                    try: self.models[doc_type] = get_model("Document").objects.doc2vec(doc_type)
                    except EOFError:
                        time.sleep(1)
                        self.models[doc_type] = get_model("Document").objects.doc2vec(doc_type)
                        if self.models[doc_type] == None:
                            print "Couldn't load doc2vec model!"
                            raise Exception
                    self.set_row_cache("model", self.models[doc_type])

        return self

    def get_feature_names(self):

        prefix = "{0}_".format(self.params["feature_name_prefix"]) if self.params["feature_name_prefix"] else ""
        feature_names = []
        for doc_type, model in self.models.iteritems():
            feature_names.extend(
                ["{0}{1}_{2}".format(prefix, doc_type, i) for i in range(0, model.vector_size)]
            )
        return feature_names