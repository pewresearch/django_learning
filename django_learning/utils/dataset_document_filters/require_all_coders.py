from __future__ import absolute_import


def filter(self, df):
    grouped = df.groupby("document_id").agg({"coder_id": lambda x: len(set(x))})[
        "coder_id"
    ]
    return df[df["document_id"].isin(grouped[grouped == grouped.max()].index)]
