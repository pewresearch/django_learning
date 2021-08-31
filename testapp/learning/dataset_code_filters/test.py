from __future__ import absolute_import

from django_pewtils import get_model


def filter(self, df, *args, **kwargs):
    pk = (
        get_model("Question", app_name="django_learning")
        .objects.get(name="test_checkbox")
        .labels.get(value="1")
        .pk
    )
    return df[df["label_id"] == pk]
