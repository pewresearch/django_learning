

def filter(documents, min_date=None, max_date=None):

    if min_date:
        documents = documents.filter(date__date__gte=min_date)
    if max_date:
        documents = documents.filter(date__date__lte=max_date)

    return documents