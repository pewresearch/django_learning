from __future__ import print_function

from builtins import zip
from builtins import str

from django.conf import settings

from logos.models import *
from logos.utils import is_not_null, is_null
from logos.utils.data import wmom

from collections import defaultdict
from nltk.metrics.agreement import AnnotationTask
from statsmodels.stats.inter_rater import cohens_kappa
from django.db.models import Q
from contextlib import closing

import pandas, datetime
from django.core.management.base import BaseCommand, CommandError

from logos.models import *

import pandas

from logos.models import *
from logos.utils import is_not_null, is_null
from logos.utils.data import wmom
from logos.utils.io import FileHandler

from collections import defaultdict
from nltk.metrics.agreement import AnnotationTask
from statsmodels.stats.inter_rater import cohens_kappa
from django.db.models import Q
from contextlib import closing


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        pass

    def handle(self, *args, **options):

        mean_results = defaultdict(list)
        for c in CodeVariableClassifier.objects.all():

            print(c.name)
            h = c.handler
            h.load_model()

            dfs = {}
            dfs["expert"] = h._get_expert_consensus_dataframe("1", coders=["pvankessel", "ahughes"])
            # patrick_df = h._get_expert_consensus_dataframe(self, code_value, coders=["pvankessel"])
            # adam_df = h._get_expert_consensus_dataframe(self, code_value, coders=["ahughes"])
            code_map = {int(code['pk']): int(code['value']) for code in c.variable.codes.values("pk", "value")}
            dfs["turk"] = h.test_x
            dfs["turk"]['code_id'] = h.test_y
            dfs["turk"]['code'] = dfs["turk"]['code_id'].map(code_map)
            dfs["turk_full"] = h.train_x
            dfs["turk_full"]['code_id'] = h.train_y
            dfs["turk_full"]['code'] = dfs["turk_full"]['code_id'].map(code_map)

            for dfname, df in dfs.items():
                if is_not_null(df):
                    if "press_release" in c.document_types:
                        dfs[dfname]['party'] = dfs[dfname].apply(
                            lambda x: Document.objects.get(pk=x['document_id']).press_release.politician.party,
                            axis=1)
                    elif "facebook_post" in c.document_types:
                        dfs[dfname]['party'] = dfs[dfname].apply(
                            lambda x: Document.objects.get(pk=x['document_id']).facebook_post.source_page.politician.party,
                            axis=1)
                        dfs[dfname]['is_official'] = dfs[dfname].apply(
                            lambda x: Document.objects.get(pk=x['document_id']).facebook_post.source_page.is_official,
                            axis=1)
                        dfs[dfname] = dfs[dfname][dfs[dfname]['is_official'] == True]
                    dfs[dfname].ix[dfs[dfname]['party'].isnull(), 'party'] = "D"
                    dfs[dfname]["party"] = dfs[dfname]["party"].map(lambda x: str(x)[0])

            if len(dfs['turk_full']['party'].unique()) == 1:
                parties = dfs['turk_full']['party'].unique()
            else:
                parties = [None, "R", "D"]
            for party in parties:
                print("{}, {}".format(c.name, party))
                result = {
                    "doc_type": c.document_types[0],
                    "var": c.variable.name,
                    "party": party
                }
                for dfname, df in dfs.items():
                    if is_not_null(df):
                        if is_not_null(party):
                            df = df[df['party'] == party]
                        result['num_docs_{}'.format(dfname)] = len(df)
                        unweighted = wmom(df['code'].values, [1.0 for v in df['code'].values], calcerr=True,
                                          sdev=True)
                        for valname, val in zip(["mean", "err", "std"], list(unweighted)):
                            result["{}_{}".format(dfname, valname)] = val
                        weighted = wmom(df['code'].values, df['sampling_weight'], calcerr=True, sdev=True)
                        for valname, val in zip(["mean", "err", "std"], list(weighted)):
                            result["{}_{}_weighted".format(dfname, valname)] = val
                # for pairname, y1, y2 in [
                #     ("turk_expert", dfs['turk'], dfs['expert']),
                # ]:
                #     if is_not_null(y1) and is_not_null(y2):
                #         if is_not_null(party):
                #             y1 = y1[y1['party'] == party]
                #             y2 = y2[y2['party'] == party]
                #         intersect_ids = set(y1["document_id"].values).intersection(
                #             set(y2['document_id'].values))
                #         y1 = y1[y1['document_id'].isin(intersect_ids)]
                #         y2 = y2[y2['document_id'].isin(intersect_ids)]
                #         y1 = y1['code'].values
                #         y2 = y2['code'].values
                #         result_dict = {}
                #         vals = list(set(y1))
                #         if len(vals) == 2:
                #             for v in vals: result_dict[v] = defaultdict(int)
                #             for turk, model in zip(y1, y2):
                #                 result_dict[turk][model] += 1
                #             kappa = cohens_kappa([
                #                 [result_dict.get(vals[0], {}).get(vals[0], 0),
                #                  result_dict.get(vals[0], {}).get(vals[1], 0)],
                #                 [result_dict.get(vals[1], {}).get(vals[0], 0),
                #                  result_dict.get(vals[1], {}).get(vals[1], 0)]
                #             ])
                #             result["{}_kappa".format(pairname)] = kappa["kappa"]
                #             result["{}_kappa_std".format(pairname)] = kappa["std_kappa"]
                mean_results[(c.variable.name, c.document_types[0])].append(result)

        results = []
        for rset in list(mean_results.values()):
            results.extend(rset)
        df = pandas.DataFrame(results)
        df = df[["doc_type", "var", "party"] + [c for c in df.columns if "mean" in c]]
        df["party"] = df["party"].map(lambda x: x if x else "All")

        h = FileHandler("output/queries/partisan_antipathy",
            use_s3=True,
            bucket=settings.S3_BUCKET,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY
        )
        h.write("mean_results_by_party", df, format="csv")