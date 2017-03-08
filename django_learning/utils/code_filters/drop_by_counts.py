
def filter(codes, min_docs=10, **kwargs):

    valid_coders = []
    for coder_id, group in codes.groupby("coder_id"):
        if len(group["document_id"].unique()) > min_docs:
            valid_coders.append(coder_id)
    codes = codes[codes["coder_id"].isin(valid_coders)]

    return codes