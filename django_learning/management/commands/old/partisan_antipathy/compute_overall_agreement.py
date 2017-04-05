import numpy, itertools, pandas, copy

from logos.learning.supervised import DocumentClassificationHandler
from logos.models import *
from logos.utils import is_not_null, is_null
from logos.utils.data import wmom
from logos.utils.io import FileHandler

from collections import defaultdict
from nltk.metrics.agreement import AnnotationTask
from statsmodels.stats.inter_rater import cohens_kappa
from django.db.models import Q
from contextlib import closing

from logos.learning.supervised import DocumentClassificationHandler
from logos.models import *
from collections import defaultdict
from nltk.metrics.agreement import AnnotationTask

import pandas, datetime
from django.core.management.base import BaseCommand, CommandError

from logos.models import *


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("--original_consensus_subset_only", default=False, action="store_true")

    def handle(self, *args, **options):

        var_sets = [
            #     # ["oppose_obama", "oppose_dems", "oppose_reps"],
            #     # ["bipartisanship", "oppose_obama", "oppose_dems", "oppose_reps"],
            #     # ["bipartisanship", "expresses_anger", "oppose_obama", "oppose_dems", "oppose_reps"],
            #     # ["bipartisanship", "oppose_obama", "oppose_dems", "oppose_reps", "district_benefit"],
            # ["bipartisanship", "expresses_anger", "oppose_obama", "oppose_dems", "oppose_reps"],
            # ["bipartisanship", "expresses_anger", "oppose_obama", "oppose_dems", "oppose_reps", "district_benefit"],
            # ["bipartisanship", "oppose_political_figures_with_anger", "oppose_political_figures"],
            # ["bipartisanship", "oppose_political_figures"],
            # ["bipartisanship", "oppose_political_figures", "district_benefit"],
            ("facebook_post", ["bipartisanship", "oppose_political_figures_with_anger", "oppose_political_figures"]),
            ("press_release", ["bipartisanship", "oppose_political_figures_with_anger", "oppose_political_figures", "district_benefit"]),
            # ["oppose_political_figures"],
            # ["oppose_political_figures_with_anger"]
        ]

        results = []

        for doc_type, var_set in var_sets:

            print "{}, {}".format(doc_type, var_set)

            dataframes = {}
            for var in var_set:

                c = CodeVariableClassifier.objects.get(name="{}_{}".format(var, doc_type))

                # print "\n{}\n".format(c.name)
                base_row = {
                    "classifier": c.name,
                    "threshold": c.parameters["codes"].get("mturk", {}).get("consolidation_threshold", None),
                    "classifier_id": c.pk,
                    "document_types": c.document_types
                }
                # print base_row

                h = c.handler
                h.load_model()

                code_map = {int(code['pk']): int(code['value']) for code in c.variable.codes.values("pk", "value")}

                dfs = {}

                dfs['patrick'] = h._get_expert_consensus_dataframe("1", coders=["pvankessel"], use_consensus_ignore_flag=False)
                dfs['patrick'].index = dfs['patrick']['document_id']
                dfs["patrick"]['code'] = dfs["patrick"]['code_id'].map(code_map)

                dfs['adam'] = h._get_expert_consensus_dataframe("1", coders=["ahughes"], use_consensus_ignore_flag=False)
                dfs['adam'].index = dfs['adam']['document_id']
                dfs["adam"]['code'] = dfs["adam"]['code_id'].map(code_map)

                dfs["turk"] = copy.deepcopy(h.test_x)
                dfs["turk"]['code_id'] = h.test_y
                dfs["turk"]['code'] = dfs["turk"]['code_id'].map(code_map)
                dfs['turk'].index = dfs['turk']['document_id']

                dfs["expert"] = h._get_expert_consensus_dataframe("1", coders=["pvankessel", "ahughes"], use_consensus_ignore_flag=(not options["original_consensus_subset_only"]))
                dfs['expert'].index = dfs['expert']['document_id']

                # Filter out validation cases that were also included in the training sample
                h.predict_y = pandas.Series(h.predict_y, index=h.test_x.index)
                h.test_x = h.test_x[~h.test_x['document_id'].isin(h.train_x['document_id'])]
                h.test_y = h.test_y.ix[h.test_x.index]
                h.predict_y = h.predict_y[h.test_x.index].values

                dfs["model"] = copy.deepcopy(h.test_x)
                dfs["model"]["code_id"] = h.predict_y
                dfs["model"]["code"] = dfs["model"]["code_id"].map(code_map)
                dfs["model"].index = dfs["model"]["document_id"]

                base_row['num_docs'] = len(dfs['expert'])

                for dfname, df in dfs.iteritems():
                    if is_not_null(df):
                        if "press_release" in c.document_types:
                            dfs[dfname]['party'] = dfs[dfname].apply(
                                lambda x: Document.objects.get(pk=x['document_id']).press_release.politician.party,
                                axis=1)
                        elif "facebook_post" in c.document_types:
                            dfs[dfname]['party'] = dfs[dfname].apply(lambda x: Document.objects.get(
                                pk=x['document_id']).facebook_post.source_page.politician.party, axis=1)
                        dfs[dfname][dfs[dfname]['party'].isnull()]['party'] = "D"
                        dfs[dfname]["party"] = dfs[dfname]["party"].map(lambda x: str(x)[0])

                dataframes[var] = dfs

            master_df = {}
            for var, dfs in dataframes.iteritems():
                for dfname, df in dfs.iteritems():
                    df = df[["document_id", "code"]]
                    df = df.rename(columns={"code": var})
                    if dfname not in master_df.keys():
                        master_df[dfname] = df
                    else:
                        master_df[dfname] = pandas.merge(master_df[dfname], df, how='outer', on="document_id")

            for dfname, df in master_df.iteritems():
                master_df[dfname]['allNA'] = master_df[dfname].apply(lambda x: 1 if all([pandas.isnull(x[var]) for var in var_set]) else 0, axis=1)
                master_df[dfname] = master_df[dfname][master_df[dfname]['allNA']==0].fillna(0)
                del master_df[dfname]['allNA']
                master_df[dfname][dfname] = master_df[dfname].apply(lambda x: "".join(str(int(x[var])) for var in var_set), axis=1)

            result = {"var_set": var_set, "doc_type": doc_type}
            for pairname, df1, df2 in [
                ("expert_expert", "patrick", "adam"),
                ("turk_expert", "expert", "turk"),
                ("model_turk", "model", "turk"),
                ("model_expert", "model", "expert")
            ]:
                index_intersect = master_df[df1].index.intersection(master_df[df2].index)
                rows = []
                pairdf = pandas.merge(master_df[df1][["document_id", df1]], master_df[df2][["document_id", df2]], how="inner", on="document_id")
                for index, row in pairdf.iterrows():
                    rows.append([1, int(row["document_id"]), row[df1]])
                    rows.append([2, int(row["document_id"]), row[df2]])
                task = AnnotationTask(data=rows)
                kappa = task.kappa()
                print "{} kappa: {} ({} rows)".format(pairname, kappa, len(index_intersect))
                result[pairname] = kappa
            results.append(result)

        results = pandas.DataFrame(results)
        print results

        if options["original_consensus_subset_only"]:
            suffix = "_consensus_subset"
        else:
            suffix = ""
        h = FileHandler("output/queries/partisan_antipathy", use_s3=True)
        h.write("combined_kappas{}".format(suffix), results, format="csv")