def filter(self, df, min_date=None, max_date=None):

    if min_date:
        df = df[df["date"] >= min_date]
    if max_date:
        df = df[df["date"] <= max_date]

    return df
