from django.db.models import Q

from pewtils import is_not_null
from django_pewtils import get_model


def filter(
    self, df, search_filter=None, search_value=None, exclude=False, operator=None
):

    documents = get_model("Document", app_name="django_learning").objects.filter(
        pk__in=df["document_id"]
    )
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

    return df[df["document_id"].isin(list(documents.values_list("pk", flat=True)))]
