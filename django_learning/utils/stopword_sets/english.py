from __future__ import absolute_import

from sklearn.feature_extraction import text as sklearn_text


def get_stopwords():

    return list(sklearn_text.ENGLISH_STOP_WORDS)
