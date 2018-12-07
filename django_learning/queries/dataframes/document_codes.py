import pandas

from tqdm import tqdm

from django_pewtils import get_model
from pewtils import is_not_null
from django_queries.queries.dataframes import DataFrameQuery


class DataFrame(DataFrameQuery):

    model = "Document"

    parameter_defaults = [
        {"name": "min_date", "default": None, "help": ""},
        {"name": "max_date", "default": None, "help": ""},
        {"name": "document_type", "default": None, "help": ""},
        {"name": "classifier_name", "default": None, "help": ""}
    ]

    option_defaults = [
        {"name": "refresh", "default": False, "help": ""}
    ]

    def _extract_dataframe(self):

        docs = self.queryset
        if "document_type" in self.parameters.keys(): docs = docs.filter(**{"{}__isnull".format(self.parameters["document_type"]): False})
        if "min_date" in self.parameters.keys() and is_not_null(self.parameters["min_date"]): docs = docs.filter(date__gte=self.parameters["min_date"])
        if "max_date" in self.parameters.keys() and is_not_null(self.parameters["max_date"]): docs = docs.filter(date__lte=self.parameters["max_date"])

        codes = get_model("ClassifierDocumentCode", app_name="django_learning").objects.filter(document__in=docs).values("document_id", "code__value", "code__variable__name")
        if "classifier_name" in self.parameters.keys() and is_not_null(self.parameters["classifier_name"]):
            codes = codes.filter(classifier__name=self.parameters["classifier_name"])
        df = pandas.DataFrame.from_records(codes)
        df = pandas.pivot_table(df, index="document_id", columns="code__variable__name", values="code__value", aggfunc="max")

        return df