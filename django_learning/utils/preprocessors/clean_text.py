from __future__ import print_function
import re

from pewtils import is_not_null, decode_text
from pewanalytics.text import TextCleaner, SentenceTokenizer
from pewanalytics.internal.stopwords import is_probable_stopword

from django_learning.utils.stopword_sets import stopword_sets
from django_learning.utils.stopword_whitelists import stopword_whitelists
from django_learning.utils.regex_filters import regex_filters
from django_learning.utils.regex_replacers import regex_replacers
from django_learning.utils.preprocessors import BasicPreprocessor


class Preprocessor(BasicPreprocessor):

    def __init__(self, *args, **kwargs):

        self.name = "clean_text"
        super(Preprocessor, self).__init__(*args, **kwargs)

        whitelist = []
        if "stopword_whitelists" in self.params.keys():
            for stopword_whitelist in self.params["stopword_whitelists"]:
                whitelist.extend(stopword_whitelists[stopword_whitelist]())
        stopwords = []
        if "stopword_sets" in self.params.keys():
            for stopword_set in self.params['stopword_sets']:
                slist = None
                if self.cache and not self.params.get("refresh_stopwords", False):
                    slist = self.cache.read(stopword_set)
                if not slist:
                    slist = stopword_sets[stopword_set]()
                    slist = [decode_text(s) for s in slist if len(s) > 2 or stopword_set in ["english", "months", "misc_boilerplate"]]
                    if stopword_set not in ["english", "months", "misc_boilerplate"]:
                        final_slist = []
                        for s in slist:
                            s = s.lower()
                            if len(s) > 3 or is_probable_stopword(s) or self.params.get("override_stopword_check", False):
                                final_slist.append(s)
                        slist = final_slist
                    if self.cache:
                        print("Recomputed stopword set {}, saving to local cache".format(stopword_set))
                        self.cache.write(stopword_set, slist)
                stopwords.extend(slist)
        stopwords = list(set(stopwords))
        stopwords = [s for s in stopwords if s not in whitelist]
        stopwords = sorted(stopwords, key=lambda x: len(x), reverse=True)
        self.stopwords = stopwords

        replacers = []
        for r in self.params.get("regex_replacers", []):
            replacers.extend(regex_replacers[r]())
        kwargs = {"decode_text": True, "stopwords": stopwords, "strip_html": True, "replacers": replacers}
        kwargs.update({k: v for k, v in self.params.items() if k not in ["regex_replacers", "stopword_sets", "regex_filters",
                                                                         "cache_identifier", "stopword_whitelists", "refresh_stopwords",
                                                                         "override_stopword_check"]})
        self.cleaner = TextCleaner(**kwargs)
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