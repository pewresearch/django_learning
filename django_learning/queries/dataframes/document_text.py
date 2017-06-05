import pandas

from django.db.models import F
from tqdm import tqdm

from django_queries.queries.dataframes import DataFrameQuery
from django_learning.utils.preprocessors.clean_text import Preprocessor as TextCleaner


class DataFrame(DataFrameQuery):

    model = "Document"

    parameter_defaults = []

    option_defaults = [
        {"name": "refresh", "default": False, "help": ""}
    ]

    def _extract_dataframe(self):

        c = TextCleaner(**{
            "lemmatize": True,
            "regex_filters": [],
            "stopword_sets": ["english", "politicians", "states", "locations", "months", "misc_boilerplate"],
            "strip_html": True
        })

        df = pandas.DataFrame.from_records(
            self.queryset.values("pk", "date", "text")
        )
        df.rename(columns={"pk": "document_id"})
        df['text_clean'] = df['text'].map(c.run)

        return df