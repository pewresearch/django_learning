from django_pewtils import get_model


def var_mapper(x):

    doc = get_model("Document", app_name="django_learning").objects.get(
        pk=x["document_id"]
    )
    if doc.date and doc.date.month and doc.date.year:
        return "{}_{}".format(doc.date.year, doc.date.month)
    else:
        return None
