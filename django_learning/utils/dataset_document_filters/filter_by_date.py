def filter(self, df, min_date=None, max_date=None):

    if min_date:
        try:
            df = df[df["date"] >= min_date]
        except TypeError:
            ## TypeError arises 'naturally' if date is stored as
            ## `datetime64[ns]` (you know, pandas) and arguments are
            ## passed as `date`. In this case, extracting dt.date
            ## recovers a `date` to make the comparison valid.
            df = df[df["date"].dt.date >= min_date]
    if max_date:
        try:
            df = df[df["date"] <= max_date]
        except TypeError:
            df = df[df["date"].dt.date <= max_date]

    return df
