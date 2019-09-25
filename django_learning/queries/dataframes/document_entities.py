from __future__ import print_function
import pandas

from django.db.models import Count
from tqdm import tqdm

from django_pewtils import get_model
from django_queries.queries.dataframes import DataFrameQuery


class DataFrame(DataFrameQuery):

    model = "Document"

    parameter_defaults = [
        {"name": "min_date", "default": None, "help": ""},
        {"name": "max_date", "default": None, "help": ""},
        {"name": "document_type", "default": None, "help": ""},
        {"name": "top_n", "default": 1000, "help": ""},
    ]

    option_defaults = [{"name": "refresh", "default": False, "help": ""}]

    def _extract_dataframe(self):

        print("Extracting top 1000 entities")

        docs = self.queryset
        if "document_type" in self.parameters.keys():
            docs = docs.filter(
                **{"{}__isnull".format(self.parameters["document_type"]): False}
            )
        if "min_date" in self.parameters.keys():
            docs = docs.filter(date__gte=self.parameters["min_date"])
        if "max_date" in self.parameters.keys():
            docs = docs.filter(date__lte=self.parameters["max_date"])
        entities = (
            docs.values("entities__pk")
            .annotate(c=Count("pk"))
            .order_by("-c")[: self.parameters["top_n"]]
        )
        entities = get_model("Entity", app_name="django_learning").objects.filter(
            pk__in=[e["entities__pk"] for e in entities]
        )

        rows = []
        for d in tqdm(
            docs,
            desc="Iterating over documents and extracting entity flags",
            total=docs.count(),
        ):
            row = {"pk": d.pk}
            for e in entities:
                row["entity_{}".format(e.name)] = 1 if e in d.entities.all() else 0

        return pandas.DataFrame.from_records(rows)
