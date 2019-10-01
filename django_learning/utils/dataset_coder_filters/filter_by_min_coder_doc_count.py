from __future__ import absolute_import


def filter(self, df, min_docs=10):
    valid_coders = []
    for coder_id, group in df.groupby("coder_id"):
        if len(group["document_id"].unique()) > min_docs:
            valid_coders.append(coder_id)
    codes = df[df["coder_id"].isin(valid_coders)]

    return codes
