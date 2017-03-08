from django.db.models import Q

from pewtils import is_not_null


def filter(documents, search_filter=None, search_value=None, exclude=False, operator=None):

    if search_filter and search_value:
        if type(search_filter) == list and is_not_null(operator):
            query = Q()
            for f in search_filter:
                query.add(Q(**{f: search_value}), operator)
            if exclude:
                documents = documents.exclude(query)
            else:
                documents = documents.filter(query)
        else:
            if exclude:
                documents = documents.exclude(**{search_filter: search_value})
            else:
                documents = documents.filter(**{search_filter: search_value})

    return documents