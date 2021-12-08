def filter(self, df, min_date=None, max_date=None):

    if min_date:
        try:
            df = df[df["date"] >= min_date]
        except TypeError:
            df = df[df["date"].dt.date >= min_date]
    if max_date:
        try:
            df = df[df["date"] <= max_date]
        except TypeError:
            df = df[df["date"].dt.date <= max_date]

    return df
