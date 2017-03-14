import re, importlib

from nltk.corpus import wordnet

from pewtils import is_not_null, decode_text
from pewtils.nlp import TextCleaner, SentenceTokenizer, is_probable_stopword
from pewtils.django import CacheHandler

from django_learning.utils import stopword_sets, regex_filters
from django_learning.utils.preprocessors import BasicPreprocessor


WHITELIST = [
    "state university",
    "university",
    "former",
    "for president",
    "call",
    "bottom",
    "cut off",
    "fair",
    "canada",
    "market",
    "bolivia",
    "sincere",
    "9/11",
    "should",
    "must",
    "need",
    "needs",
    "onset",
    "courage",
    "new home",
    "hometown",
    "turkey",
    "sudan",
    "underage",
    "isis",
    "flint",
    "new roads",
    "wheat",
    "gateway",
    "patriot",
    "israel",
    "friendship",
    "severance",
    "system",
    "together",
    "vice",
    "jesus",
    "god",
    "interest",
	"iran",
	"syria",
	"interest",
	"never",
	"cannot",
	"not",
	"yucca",
    "capitol",
    "prosperity",
    "homeland",
    "bureau",
    "student",
    "serious",
    "century",
    "bedrock",
    "republic",
	"full",
	"against",
	"never",
    "equality",
    "fire",
    "against",
    "dairy",
	"interests",
	"hurricane",
	"op-ed",
	"cliff",
	"against"
]

class Preprocessor(BasicPreprocessor):

    def __init__(self, *args, **kwargs):

        self.name = "clean_text"
        super(Preprocessor, self).__init__(*args, **kwargs)

        stopwords = []
        if "stopword_sets" in self.params.keys():
            for stopword_set in self.params['stopword_sets']:
                slist = None
                if self.cache:
                    slist = self.cache.read(stopword_set)
                if not slist:
                    slist = stopword_sets[stopword_set]()
                    slist = [decode_text(s) for s in slist if len(s) > 2 or stopword_set in ["english", "misc_boilerplate"]]
                    if stopword_set not in ["english", "misc_boilerplate"]:
                        final_slist = []
                        for s in slist:
                            if s.lower() not in WHITELIST:
                                if len(s) > 3: final_slist.append(s)
                                elif is_probable_stopword(s.lower()):
                                    final_slist.append(s)
                        slist = final_slist
                    if self.cache:
                        # print "Recomputed stopword set {}, saving to local cache".format(stopword_set)
                        self.cache.write(stopword_set, slist)
                stopwords.extend(slist)
        stopwords = sorted(list(set(stopwords)), key=lambda x: len(x), reverse=True)
        self.stopwords = stopwords

        self.cleaner = TextCleaner(
            skip_decode=True,
            stopwords=stopwords,
            **{k: v for k, v in self.params.items() if k not in ["stopword_sets", "regex_filters", "cache_identifier"]}
        )
        self.tokenizer = SentenceTokenizer()

        self.regex_filters = []
        if "regex_filters" in self.params.keys():
            for regex_filter in self.params['regex_filters']:
                self.regex_filters.append(regex_filters[regex_filter]())

        self.url_regex = re.compile(
            r"((https?:\/\/(www\.)?)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))"
        )

    def run(self, text):

        key_text = text
        clean_text = None
        if self.cache:
            clean_text = self.get_row_cache(key_text)
            if clean_text:
                pass
                # print "Loaded from cache"

        if not clean_text:

            text = " {} ".format(decode_text(text))
            for filter in self.regex_filters:
                final_tokens = []
                for sent in self.tokenizer.tokenize(text):
                    if is_not_null(sent) and filter.search(sent):
                        final_tokens.append(sent)
                text = " ".join(final_tokens)
                    # sent = self.cleaner.clean(sent)
                    # if self.re_search.match(sent):
                    #     final_tokens.append(sent)
                # tokens = self.cleaner.clean(sent).split()
                # for i, token in enumerate(tokens):
                #     if self.re_search.match(token):
                #         final_tokens.extend(tokens[max([0, i-self.params['token_window_size']]):min([len(tokens), i+self.params['token_window_size']])])

            # found_stopwords = self.stopword_regex.findall(text)
            # text = self.stopword_regex.sub('', text)
            clean_text = self.cleaner.clean(text)
            # print text
            # print "Stopwords removed: {}".format(set(found_stopwords))
            # probable_stopwords = set()
            # for w in text.split():
            #     if is_probable_stopword(w):
            #         probable_stopwords.add(w)
            # print probable_stopwords

            if self.cache:
                self.set_row_cache(key_text, clean_text)

        return clean_text