from __future__ import absolute_import


def filter(self, df, *args, **kwargs):

    return df[df["label_id"] == 12]
