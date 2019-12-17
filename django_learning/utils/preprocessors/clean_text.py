from __future__ import print_function
import re

from pewtils import is_not_null, decode_text
from pewanalytics.text import TextCleaner, SentenceTokenizer

from django_learning.utils.stopword_sets import stopword_sets
from django_learning.utils.stopword_whitelists import stopword_whitelists
from django_learning.utils.regex_filters import regex_filters
from django_learning.utils.regex_replacers import regex_replacers
from django_learning.utils.preprocessors import BasicPreprocessor


class Preprocessor(BasicPreprocessor):
    def __init__(self, *args, **kwargs):
        """

        :param stopword_sets: Names of `django_learning` stopword lists
        :param stopword_whitelists: Names of `django_learning` stopword whitelists (used to override stopword_sets)
        :param regex_filters: Names of `django_learning` regex filters; sentences that don't match to *all* of the
        provided filters will be removed
        :param regex_replacers: Names of `django_learning` regex replacers to use
        :param refresh_stopwords: Whether or not to refresh a custom list of stopwords from the cache (default is False)
        :param kwargs: All other keyword arguments are passed along to a `pewanalytics.text.TextCleaner` object

        """

        self.name = "clean_text"
        super(Preprocessor, self).__init__(*args, **kwargs)

        whitelist = []
        if "stopword_whitelists" in self.params.keys():
            for stopword_whitelist in self.params["stopword_whitelists"]:
                whitelist.extend(stopword_whitelists[stopword_whitelist]())
        stopwords = []
        if "stopword_sets" in self.params.keys():
            for stopword_set in self.params["stopword_sets"]:
                slist = None
                if self.cache and not self.params.get("refresh_stopwords", False):
                    slist = self.cache.read(stopword_set)
                if not slist:
                    slist = stopword_sets[stopword_set]()
                    slist = [
                        decode_text(s)
                        for s in slist
                        if len(s) > 2
                        or stopword_set in ["english", "months", "misc_boilerplate"]
                    ]
                    if self.cache:
                        print(
                            "Recomputed stopword set {}, saving to local cache".format(
                                stopword_set
                            )
                        )
                        self.cache.write(stopword_set, slist)
                stopwords.extend(slist)
        stopwords = list(set(stopwords))
        stopwords = [s for s in stopwords if s not in whitelist]
        stopwords = sorted(stopwords, key=lambda x: len(x), reverse=True)
        self.stopwords = stopwords

        replacers = []
        for r in self.params.get("regex_replacers", []):
            replacers.extend(regex_replacers[r]())
        kwargs = {"stopwords": stopwords, "strip_html": True, "replacers": replacers}
        kwargs.update(
            {
                k: v
                for k, v in self.params.items()
                if k
                not in [
                    "regex_replacers",
                    "stopword_sets",
                    "regex_filters",
                    "cache_identifier",
                    "stopword_whitelists",
                    "refresh_stopwords",
                ]
            }
        )
        self.cleaner = TextCleaner(**kwargs)
        self.tokenizer = SentenceTokenizer()

        self.regex_filters = []
        if "regex_filters" in self.params.keys():
            for regex_filter in self.params["regex_filters"]:
                self.regex_filters.append(regex_filters[regex_filter]())

        self.url_regex = re.compile(
            r"((https?:\/\/(www\.)?)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*))"
        )

    def run(self, text):

        key_text = text
        clean_text = None
        if self.cache:
            clean_text = self.get_row_cache(key_text)

        if not clean_text:

            text = " {} ".format(decode_text(text))
            for filter in self.regex_filters:
                final_tokens = []
                for sent in self.tokenizer.tokenize(text):
                    if is_not_null(sent) and filter.search(sent):
                        final_tokens.append(sent)
                text = " ".join(final_tokens)

            clean_text = self.cleaner.clean(text)

            if self.cache:
                self.set_row_cache(key_text, clean_text)

        return clean_text
