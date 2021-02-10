import nltk
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def get_stopwords():

    return list(
        set.union(set(nltk.corpus.stopwords.words("english")), set(ENGLISH_STOP_WORDS))
    )
