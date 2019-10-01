from __future__ import absolute_import


def filter(self, df):
    return df[df["coder_is_mturk"]]
