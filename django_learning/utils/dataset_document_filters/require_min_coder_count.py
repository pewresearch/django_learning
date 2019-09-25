def filter(self, df, min_count):
    counts = df.groupby("document_id").count()
    return df[df["document_id"].isin(counts[counts["coder_id"] >= min_count].index)]
