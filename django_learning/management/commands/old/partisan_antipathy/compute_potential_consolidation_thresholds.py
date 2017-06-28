import pandas, math, re, numpy, cPickle, copy

from django.conf import settings

from logos.learning.supervised import DocumentClassificationHandler
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


class Command(BaseCommand):
    """
    """

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("--original_consensus_subset_only", default=False, action="store_true")

    def handle(self, *args, **options):

        from logos.utils.io import FileHandler
        h = FileHandler("output/queries/partisan_antipathy",
            use_s3=True,
            bucket=settings.S3_BUCKET,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY
        )

        var_names = [
            "district_benefit",
            "bipartisanship",
            "expresses_anger",
            "oppose_obama",
            "oppose_dems",
            "oppose_reps"
        ]

        def get_var_df(var, sample_ids, turk=False, coder=None, exclude_filter=None):

            codes = CoderDocumentCode.objects.filter(sample_unit__sample_id__in=sample_ids).filter(coder__is_mturk=turk)
            if coder:
                codes = codes.filter(coder__name=coder)
            elif not turk:
                codes = codes.filter(coder__name__in=["ahughes", "pvankessel"])
            if not options["original_consensus_subset_only"]:
                codes = codes.exclude(consensus_ignore=True)
            if exclude_filter: codes = codes.exclude(exclude_filter)
            df = pandas.DataFrame.from_records(
                codes.values("sample_unit__document_id", "code__variable__name", "code__value", "sample_unit__weight"))
            df = df.rename(columns={"code__value": "code", "code__variable__name": "question_name",
                                    "sample_unit__document_id": "external_id", "sample_unit__weight": "weight"})
            df = df[df["question_name"] == var][["code", "external_id", "weight"]]
            df["code"] = df["code"].map(lambda x: int(x))
            df = df.groupby("external_id").mean().reset_index().sort("external_id")

            return df

        print "Computing Kappa thresholds"

        alphas = []
        for doc_type in ["facebook_post", "press_release"]:

            if doc_type == "facebook_post":
                validation_sample_ids = [12]
            elif doc_type == "press_release":
                validation_sample_ids = [22]

            # validation_sample_ids = [12, 22]

            for var_name in var_names:

                print "{}, {}".format(doc_type, var_name)

                if "_obama" in var_name or "_dems" in var_name:
                    exclude_filter = Q(
                        sample_unit__document__press_release__politician__party__name="Democratic Party") | Q(
                        sample_unit__document__facebook_post__source_page__politician__party__name="Democratic Party")
                elif "_reps" in var_name:
                    exclude_filter = Q(
                        sample_unit__document__press_release__politician__party__name="Republican Party") | Q(
                        sample_unit__document__facebook_post__source_page__politician__party__name="Republican Party")
                else:
                    exclude_filter = None

                # In-house Kappa computation

                y_patrick = get_var_df(var_name, validation_sample_ids, turk=False, coder="pvankessel",
                                       exclude_filter=exclude_filter)
                y_adam = get_var_df(var_name, validation_sample_ids, turk=False, coder="ahughes",
                                    exclude_filter=exclude_filter)

                if len(y_patrick) > 0 and len(y_adam) > 0:

                    y_patrick = y_patrick['code'].values
                    y_adam = y_adam['code'].values

                    result_dict = {0: defaultdict(int), 1: defaultdict(int)}
                    for patrick, adam in zip(y_patrick, y_adam):
                        result_dict[patrick][adam] += 1
                    expert_kappa = cohens_kappa([
                        [result_dict[0][0], result_dict[0][1]],
                        [result_dict[1][0], result_dict[1][1]]
                    ])
                    expert_kappa_err = expert_kappa["std_kappa"]
                    expert_kappa = expert_kappa["kappa"]

                    # Turk vs. Expert threshold comparisons

                    df_turk = get_var_df(var_name, validation_sample_ids, turk=True, exclude_filter=exclude_filter)
                    df_turk.index = df_turk['external_id'].values
                    df_expert = get_var_df(var_name, validation_sample_ids, turk=False, exclude_filter=exclude_filter)
                    df_expert.index = df_expert['external_id'].values

                    df_expert = df_expert[df_expert['code'].isin([1.0, 0.0])]

                    index_intersect = df_expert.index.intersection(df_turk.index)
                    df_turk = df_turk.ix[index_intersect]
                    df_expert = df_expert.ix[index_intersect]

                    y_expert_prev = None
                    y_turk_prev = None
                    # for expert_threshold in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
                    for expert_threshold in [0.0]:
                        # y_expert = df_expert['code'].map(lambda x: 1 if x >= expert_threshold else 0)
                        y_expert = df_expert['code'].map(lambda x: int(x))
                        if is_null(y_expert_prev) or list(y_expert.values) != list(y_expert_prev.values):
                            y_expert_prev = y_expert
                            for turk_threshold in [.1, .2, .3, .4, .5, .6, .7, .8, .9]:
                                y_turk = df_turk['code'].map(lambda x: 1 if x >= turk_threshold else 0)
                                if is_null(y_turk_prev) or list(y_turk.values) != list(y_turk_prev.values):
                                    y_turk_prev = y_turk
                                    if sum(y_turk) > 0 and sum(y_expert) > 0:

                                        result_dict = {0: defaultdict(int), 1: defaultdict(int)}
                                        for turk, expert in zip(y_turk, y_expert):
                                            result_dict[turk][expert] += 1

                                        kappa = cohens_kappa([
                                            [result_dict[0][0], result_dict[0][1]],
                                            [result_dict[1][0], result_dict[1][1]]
                                        ])
                                        result = {
                                            "doc_type": doc_type,
                                            "var": var_name,
                                            "num_docs": len(df_expert),
                                            "expert_threshold": expert_threshold,
                                            "turk_threshold": turk_threshold,
                                            "kappa": kappa["kappa"],
                                            "kappa_err": kappa["std_kappa"],
                                            "expert_flags": sum(y_expert),
                                            "turk_flags": sum(y_turk),
                                            "expert_kappa": expert_kappa,
                                            "expert_kappa_err": expert_kappa_err
                                        }
                                        alphas.append(result)

        table = []
        for var, vargroup in pandas.DataFrame(alphas).groupby("var"):
            var_row = {"var": var}
            for doctype, results in vargroup.groupby("doc_type"):
                candidates = []
                best_row = None
                for i, row in results.sort(["turk_threshold", "expert_threshold"], ascending=True).sort("kappa",
                                                                                                        ascending=False).iterrows():
                    if is_null(best_row):
                        best_row = row
                        candidates.append((row['kappa'], row['turk_threshold'], row['expert_threshold']))
                    elif row['kappa'] >= (best_row['kappa'] - best_row['kappa_err']):
                        candidates.append((row['kappa'], row['turk_threshold'], row['expert_threshold']))
                print "{}, {} candidates: {}".format(doctype, var, candidates)
                # print "{}, potential turk thresholds: {}".format(var, set([c[1] for c in candidates]))
                var_row["{}_num_docs".format(doctype)] = results["num_docs"].mean()
                var_row["{}_expert_threshold".format(doctype)] = "{}".format(best_row["expert_threshold"])
                var_row["{}_turk_threshold".format(doctype)] = "{}".format(best_row["turk_threshold"])
                var_row["{}_kappa".format(doctype)] = best_row["kappa"]
                var_row["{}_min_support".format(doctype)] = min([best_row["expert_flags"], best_row["turk_flags"]])
                var_row["{}_kappa_err".format(doctype)] = best_row["kappa_err"]
                var_row["{}_expert_kappa".format(doctype)] = best_row["expert_kappa"]
                var_row["{}_expert_kappa_err".format(doctype)] = best_row["expert_kappa_err"]
                var_row["{}_kappa_expert_diff".format(doctype)] = var_row["{}_kappa".format(doctype)] - best_row[
                    "expert_kappa"]
                var_row["{}_kappa_err_expert_diff".format(doctype)] = var_row["{}_kappa_err".format(doctype)] - \
                                                                      best_row["expert_kappa_err"]
            table.append(var_row)
        pandas.DataFrame(table)[['var', "facebook_post_num_docs", 'facebook_post_kappa', 'facebook_post_turk_threshold',
                                 'press_release_num_docs', 'press_release_kappa', 'press_release_turk_threshold']]
        # pandas.DataFrame(table)[['var', 'facebook_post_kappa', 'facebook_post_turk_threshold', 'press_release_kappa', 'press_release_turk_threshold']]
        # pandas.DataFrame(table)[['var', 'all_kappa', 'all_turk_threshold', 'all_expert_threshold']]

        threshold_df = pandas.DataFrame(alphas)