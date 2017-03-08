import pandas

from tqdm import tqdm

def filter(codes, **kwargs):

    if len(codes["coder_id"].unique()) > 1:

        def _standardize(group):

            for col in group.columns:
                if col.startswith("code_") and col != "code_id" and len(group[col].unique()) > 1:
                    mean = group[col].mean()
                    std = group[col].std(ddof=1)
                    group[col] = group[col].apply(lambda x: (x - mean)/std)

            return group

        # codes = pandas.concat([_standardize(group) for _, group in tqdm(codes.groupby("document_id"), desc="Standardizing for document-level variation")])
        codes = pandas.concat([_standardize(group) for _, group in tqdm(codes.groupby("coder_id"), desc="Standardizing for coder bias")])

    return codes