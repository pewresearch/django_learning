def filter(self, df, coder_names):
    return df[df['coder_name'].isin(coder_names)]