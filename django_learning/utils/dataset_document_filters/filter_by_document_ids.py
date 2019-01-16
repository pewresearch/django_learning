

def filter(self, df, document_ids):

    return df[df['document_id'].isin(document_ids)]