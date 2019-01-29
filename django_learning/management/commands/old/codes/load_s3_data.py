from builtins import str
import boto, os, pandas

from collections import defaultdict
from django.core.management.base import BaseCommand, CommandError

from logos.models import Coder, CodeVariable
from logos.settings import LOCAL_CACHE_ROOT
from logos.utils import get_model_by_document_type

class Command(BaseCommand):
    """
    """
    help = ""

    def add_arguments(self, parser):

        parser.add_argument("--document_type", default="press_releases", type=str)
        parser.add_argument("--coder_type", default="turk", type=str)
        parser.add_argument("--s3_path", default="trustInGov/Turk_2154412_batch_results.csv", type=str)
        parser.add_argument("--refresh_data", default=False, action="store_true")

    def handle(self, *args, **options):

        """
        :param document_type: press_release ... etc.
        :param coder_type:  turk or CAT
        """
        doc_model, metadata = get_model_by_document_type(options["document_type"])
        metadata["coder_code_model"].objects.load_s3_data(
            s3_path=options["s3_path"],
            coder_type=options["coder_type"],
            refresh_data=options["refresh_data"]
        )

        keyname = options["filename"]
        filename = keyname.replace("/", "_")
        if not os.path.exists(os.path.join(LOCAL_CACHE_ROOT, filename)):

            conn = boto.connect_s3()
            bucket = conn.get_bucket("pew-lab-internal")
            key = bucket.get_key(keyname)
            key.get_contents_to_filename(os.path.join(LOCAL_CACHE_ROOT, filename))

        df = pandas.read_csv(os.path.join(LOCAL_CACHE_ROOT, filename))
        del df["Answer.comment"]
        answer_cols = [answer for answer in df.columns if answer.split('.')[0] == 'Answer']
        doc_model, doc_col = None, None
        for col in df.columns:
            if col.endswith("_id") and col != "pol_id":
                doc_model = get_model_by_document_type(col)
                if doc_model:
                    doc_col = col
                    break
        if not doc_col or not doc_model:
            raise Exception("No document ID column was found in the provided data file!")

        metadata = doc_model.objects.metadata()
        df = df[ ['HITId', 'WorkerId', 'Input.url', 'Input.pol_id', doc_col] + answer_cols ]
        df = df.dropna(subset=answer_cols, how='all')
        df[answer_cols] = df[answer_cols].fillna(0)

        coders = {}
        for coder in df["WorkerId"].unique():
            coders[coder] = Coder.objects.create_or_update(
                {"name": coder},
                {"is_mturk": True},
                search_nulls=False,
                save_nulls=False,
                empty_lists_are_null=True
            )

        codes = defaultdict(dict)
        for col in answer_cols:
            try:
                var = CodeVariable.objects.get(name=col.split(".")[-1])
                for code in var.codes.all():
                    codes[var.name][code.value] = code
            except CodeVariable.DoesNotExist: pass

        for index, row in df.iterrows():
            for col in answer_cols:
                var = col.split(".")[-1]
                try: str_code = str(int(row[col]))
                except: str_code = str(row[col])
                if str_code in list(codes[var].keys()):
                    metadata["coder_code_model"].objects.create_or_update(
                        {
                            "%s" % metadata["id_field"]: row[doc_col],
                            "coder": coders[row["WorkerId"]],
                            "code": codes[var][str_code]
                        },
                        {"hit_id": row["HITId"]},
                        return_object=False,
                        search_nulls=True,
                        save_nulls=True,
                        empty_lists_are_null=True
                    )