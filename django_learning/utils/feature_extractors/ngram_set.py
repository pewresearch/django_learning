import pandas, re

from tqdm import tqdm
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfTransformer

from pewtils import decode_text
from pewtils.django import get_model, reset_django_connection_wrapper
from django_learning.utils.feature_extractors import BasicExtractor



class Extractor(BasicExtractor):

    def __init__(self, *args, **kwargs):

        self.name = None
        self.features = []
        self.regexes = {}
        self.ngram_regexes = {}

        super(Extractor, self).__init__(*args, **kwargs)

    def transform_text(self, text, preprocessors=None):

        if not preprocessors:
            preprocessors = self.get_preprocessors()

        for p in preprocessors:
            text = p.run(text)
        ngramset_row = defaultdict(int)
        for cat in self.regexes.keys():
            matches = self.regexes[cat].findall(u" {0} ".format(text))
            if self.params["feature_name_prefix"]:
                ngramset_row["%s__%s" % (self.params["feature_name_prefix"], cat)] = len(
                    [m for m in matches if m != ''])
            else:
                ngramset_row[cat] = len([m for m in matches if m != ''])
        if self.params["include_ngrams"]:
            for word in self.ngram_regexes.keys():
                matches = self.ngram_regexes[word].findall(u" {0} ".format(text))
                if self.params["feature_name_prefix"]:
                    ngramset_row["%s__ngram__%s" % (self.params["feature_name_prefix"], word)] = len(
                        [m for m in matches if m != ''])
                else:
                    ngramset_row["ngram__%s".format(word)] = len([m for m in matches if m != ''])

        return ngramset_row

    def transform(self, X, **transform_params):

        if "include_ngrams" not in self.params.keys():
            self.params["include_ngrams"] = False

        preprocessors = self.get_preprocessors()

        ngramsets = []
        for index, row in X.iterrows():
            text = row['text']
            ngramset_row = self.transform_text(text, preprocessors=preprocessors)
            # text_length = len(text.split())
            # if text_length > 0:
            #     ngramset_row = {k: float(v) / float(text_length) for k, v in ngramset_row.items()}
            #     import pdb
            #     pdb.set_trace()
            ngramsets.append(ngramset_row)

        ngramsets = pandas.DataFrame(ngramsets)
        self.features = ngramsets.columns

        return ngramsets

    def fit(self, X, y=None, **fit_params):

        if "include_ngrams" not in self.params.keys():
            self.params["include_ngrams"] = False

        for cat in get_model("NgramSet").objects.filter(dictionary=self.params["dictionary"]):
            full_words = []
            wildcards = []
            for word in cat.words:
                if word.endswith("*"):
                    search_word = word.replace("*", "")
                    if search_word != '':
                        wildcards.append(re.escape(search_word))
                        self.ngram_regexes[re.escape(search_word)] = re.compile(
                            r"((?<=\s)" + re.escape(search_word) + "(?=\w))", re.IGNORECASE
                        )
                else:
                    search_word = word
                    if search_word != '':
                        full_words.append(re.escape(search_word))
                        self.ngram_regexes[re.escape(search_word)] = re.compile(
                            r"((?<=\s)" + re.escape(search_word) + "(?=\W))", re.IGNORECASE
                        )
            self.regexes[decode_text(cat.name)] = re.compile(
                r"((?<=\s)" + "|".join(wildcards) + "(?=\w)|" + \
                r"(?<=\s)" + "|".join(full_words) + "(?=\W))",
                re.IGNORECASE
            )

        return self

    def get_feature_names(self):

        return self.features

    def set_params(self, *args, **kwargs):

        if "dictionary" in kwargs.keys():
            self.name = kwargs["dictionary"]

        super(Extractor, self).set_params(*args, **kwargs)

        return self


# ------- The "latest version" that was new and improved, until it stopped doing its job, perhaps because of the regex optimization

# import pandas, re, copy
#
# from tqdm import tqdm
# from collections import defaultdict
#
# from sklearn.feature_extraction.text import TfidfTransformer
#
# from logos.utils.decorators import reset_django_connection_wrapper
# from logos.utils.database import get_model
# from logos.utils.text import decode_text
# from . import BasicExtractor
#
#
# # TODO: update this to accept a parameter that leverages pre-computed DocumentNgramSet data
# # if --use_database is passed and a 'pk' or 'doc_id' field is provided, do the lookup instead of computing
#
# class Extractor(BasicExtractor):
#
#     def __init__(self, *args, **kwargs):
#
#         self.name = None
#         self.features = []
#         self.regexes = {}
#         self.ngram_regexes = {}
#
#         super(Extractor, self).__init__(*args, **kwargs)
#
#     def transform(self, X, **transform_params):
#
#         if "include_ngrams" not in self.params.keys():
#             self.params["include_ngrams"] = False
#
#         X = copy.deepcopy(X)
#         for p in self.get_preprocessors():
#             X['text'] = X['text'].apply(p.run)
#
#         ngramsets = []
#         for index, row in tqdm(X.iterrows(), desc="Computing {0} categories".format(self.params["dictionary"])):
#             cache_key = "{0}_{1}.pkl".format(self.params["dictionary"], row['text'])
#             #ngramset_row = self.get_row_cache(cache_key)
#             #if not ngramset_row:
#             text = row['text']
#             # for p in preprocessors:
#             #     text = p.run(text)
#             ngramset_row = defaultdict(int)
#             # TODO: create a 'use_database' parameter that optionally attempts to pull pre-computed counts from the database
#             # db_cats = get_model("Document").objects.get(pk=row['pk']).ngram_sets.filter(ngram_set__dictionary=self.params["dictionary"]).values("ngram_set__name", "count")
#             # if db_cats.count() > 0:
#             #     for cat in db_cats:
#             #         if self.params["feature_name_prefix"]:
#             #             ngramset_row["%s__%s" % (self.params["feature_name_prefix"], cat['ngram_set__name'])] = cat['count']
#             #         else:
#             #             ngramset_row[cat['ngram_set__name']] = cat['count']
#             # else:
#             for cat in self.regexes.keys():
#                 matches = self.regexes[cat].findall(" {0} ".format(text))
#                 if self.params["feature_name_prefix"]:
#                     ngramset_row["%s__%s" % (self.params["feature_name_prefix"], cat)] = len([m for m in matches if m != ''])
#                 else:
#                     ngramset_row[cat] = len([m for m in matches if m != ''])
#             if self.params["include_ngrams"]:
#                 for word in self.ngram_regexes.keys():
#                     matches = self.ngram_regexes[word].findall(" {0} ".format(text))
#                     if self.params["feature_name_prefix"]:
#                         ngramset_row["%s__ngram__%s" % (self.params["feature_name_prefix"], word)] = len([m for m in matches if m != ''])
#                     else:
#                         ngramset_row["ngram__%s".format(word)] = len([m for m in matches if m != ''])
#             # self.set_row_cache(cache_key, ngramset_row)
#             ngramsets.append(ngramset_row)
#
#         return self.transformer.transform(
#             pandas.DataFrame(ngramsets)
#         )
#
#     def fit(self, X, y=None, **fit_params):
#
#         if "include_ngrams" not in self.params.keys():
#             self.params["include_ngrams"] = False
#
#         X = copy.deepcopy(X)
#         for p in self.get_preprocessors():
#             X['text'] = X['text'].apply(p.run)
#
#         for cat in get_model("NgramSet").objects.filter(dictionary=self.params["dictionary"]):
#             full_words = []
#             wildcards = []
#             for word in cat.words:
#                 if word.endswith("*"):
#                     search_word = word.replace("*", "")
#                     if search_word != '':
#                         search_word = re.escape(search_word)
#                         wildcards.append(search_word)
#                         # self.ngram_regexes[re.escape(search_word)] = re.compile(
#                         #     r"((?<=\s)" + re.escape(search_word) + "(?=\w))", re.IGNORECASE
#                         # )
#                         self.ngram_regexes[search_word] = re.compile(
#                             r"(\W" + search_word + "\w*\W)", re.IGNORECASE
#                         )
#                 else:
#                     search_word = word
#                     if search_word != '':
#                         search_word = re.escape(search_word)
#                         full_words.append(search_word)
#                         # self.ngram_regexes[re.escape(search_word)] = re.compile(
#                         #     r"((?<=\s)" + re.escape(search_word) + "(?=\W))", re.IGNORECASE
#                         # )
#                         self.ngram_regexes[search_word] = re.compile(
#                             r"(\W" + search_word + "\W)", re.IGNORECASE
#                         )
#             # self.regexes[decode_text(cat.name)] = re.compile(
#             #     r"((?<=\s)" + "|".join(wildcards) + "(?=\w)|" + \
#             #     r"(?<=\s)" + "|".join(full_words) + "(?=\W))",
#             #     re.IGNORECASE
#             # )
#             self.regexes[decode_text(cat.name)] = re.compile(
#                 r"\W(" + "|".join(["{}\w*".format(w) for w in wildcards]) + "|" + "|".join(full_words) + ")\W",
#                 re.IGNORECASE
#             )
#
#         # preprocessors = self.get_preprocessors()
#
#         ngramsets = []
#         for index, row in tqdm(X.iterrows(), desc="Fitting {0} categories".format(self.params["dictionary"])):
#             cache_key = "{0}_{1}.pkl".format(self.params["dictionary"], row['text'])
#             #ngramset_row = self.get_row_cache(cache_key)
#             #if not ngramset_row:
#             text = row['text']
#             # for p in preprocessors:
#             #     text = p.run(text)
#             ngramset_row = defaultdict(int)
#             for cat in self.regexes.keys():
#                 matches = self.regexes[cat].findall(" {0} ".format(text))
#                 if self.params["feature_name_prefix"]:
#                     ngramset_row["%s__%s" % (self.params["feature_name_prefix"], cat)] = len([m for m in matches if m != ''])
#                 else:
#                     ngramset_row[cat] = len([m for m in matches if m != ''])
#             if self.params["include_ngrams"]:
#                 for word in self.ngram_regexes.keys():
#                     matches = self.ngram_regexes[word].findall(" {0} ".format(text))
#                     if self.params["feature_name_prefix"]:
#                         ngramset_row["%s__ngram__%s" % (self.params["feature_name_prefix"], word)] = len([m for m in matches if m != ''])
#                     else:
#                         ngramset_row["ngram__%s".format(word)] = len([m for m in matches if m != ''])
#             # self.set_row_cache(cache_key, ngramset_row)
#             ngramsets.append(ngramset_row)
#
#         ngramsets = pandas.DataFrame(ngramsets)
#         self.features = ngramsets.columns
#
#         self.transformer = TfidfTransformer(use_idf=True)
#         self.transformer.fit(ngramsets)
#
#         return self
#
#     def get_feature_names(self):
#
#         return self.features
#
#     def set_params(self, *args, **kwargs):
#
#         if "dictionary" in kwargs.keys():
#             self.name = kwargs["dictionary"]
#
#         super(Extractor, self).set_params(*args, **kwargs)
#
#         return self


# ------- Old function that used vectorized TF-IDF columns to compute LIWC, instead of the raw text
# def transform(self, X, **transform_params):
#
#     text = X["text"]
#     if transform_params["preop"]
#
#     X = X[[c for c in X.columns if c.startswith("tfidf__")]]
#     X.columns = [c.replace("tfidf__", "") for c in X.columns]
#
#     liwcs = pandas.DataFrame()
#     ngram_cols = X.columns
#     for word in tqdm(list(self.liwc_map.keys()), desc="Aggregating ngrams into LIWC categories"):
#         if word.endswith("*"):
#             search_word = word.replace("*", "")
#             wildcard = True
#         else:
#             search_word = word
#             wildcard = False
#         regex = re.compile(r"^(" + re.escape(search_word) + ")" + ("*" if wildcard else "") + "$")
#         ngrams = filter(regex.match, ngram_cols)
#         for n in ngrams:
#             for cat in self.liwc_map[word]:
#                 liwc_col = "%s__%s" % (self.name, cat)
#                 if liwc_col in liwcs.columns:
#                     liwcs[liwc_col] += X[n]
#                 else:
#                     liwcs[liwc_col] = X[n]
#
#     # interactions = pandas.DataFrame()
#     # for ngram, liwc in tqdm(list(itertools.chain([zip(x, liwcs.columns) for x in itertools.permutations(X.columns, len(liwcs.columns))])), desc="Computing LIWC-ngram interactions"):
#     #     if ngram not in liwc_ngram_matches[liwc]:
#     #         interactions["tfidf__%s_%s" % (ngram, liwc)] = X[ngram] * liwcs[liwc]
#
#     # liwcs = pandas.concat([liwcs, interactions], ignore_index=True, axis=1)
#
#     self.features = liwcs.columns
#
#     return liwcs