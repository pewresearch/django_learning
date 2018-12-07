import pandas, time, numpy, copy

from tqdm import tqdm
from gensim.models import KeyedVectors
from django.conf import settings

from pewtils import flatten_list, is_null
from django_pewtils import get_model, reset_django_connection_wrapper
from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):

    def __init__(self, *args, **kwargs):

        self.name = "google_word2vec"
        self.w2v = None

        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        if is_null(self.w2v):
            self.w2v = self._get_w2v()

        X = copy.deepcopy(X)
        for p in self.get_preprocessors():
            X['text'] = X['text'].apply(p.run)

        full_vocab = set(self.w2v.vocab.keys())
        def mapper(x):
            words = x.split()
            words.extend(["_".join(pair) for pair in zip(words, words[1:])])
            # words = list(set(words).intersection(vocab))
            return words

        X['words'] = X['text'].map(mapper)
        sample_vocab = flatten_list(X['words'].values)
        intersect = set(sample_vocab).intersection(full_vocab)
        X['words'] = X['words'].map(lambda x: [w for w in x if w in intersect])
        X['w2v'] = X['words'].map(lambda x: [self.w2v[w] for w in x])

        rows = []
        for index, row in X.iterrows():
            doc = pandas.DataFrame(row['w2v'])
            rows.append(list(doc.mean()) + list(doc.max()) + list(doc.min()) + list(doc.std()) + list(doc.median()))

        # rows = []
        # for index, row in tqdm(X.iterrows(), desc="mapping w2v"):
        #     words = []
        #     for word in row['words']:
        #         try: wordvec = self.w2v[word]
        #         except KeyError: wordvec = None
        #         if wordvec != None:
        #             words.append(wordvec)
        #     new_row = list(pandas.DataFrame(words).mean()) + list(pandas.DataFrame(words).max()) + \
        #               list(pandas.DataFrame(words).min())
        #     rows.append(new_row)
        df = pandas.DataFrame(rows)

        self.features = df.columns

        # self.w2v = None

        return df

    def fit(self, X, y=None, **fit_params):

        if is_null(self.w2v):
            self.w2v = self._get_w2v()

        return self

    def get_feature_names(self):

        return ["{}_{}".format(self.params["feature_name_prefix"], x) for x in xrange(0, 1500)]

    def _get_w2v(self):

        return KeyedVectors.load_word2vec_format('{}/GoogleNews-vectors-negative300.bin.gz'.format(settings.FILE_ROOT), binary=True, limit=self.params["limit"])

    # def _cleanup(self):
    #
    #     self.w2v = None
    #
    #     return self
