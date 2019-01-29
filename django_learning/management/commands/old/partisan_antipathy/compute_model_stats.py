from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import zip
import pandas, math, re, numpy, pickle, copy

from django.conf import settings

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

        print("Computing mean and agreement comparison results")

        finished = []
        rows = []

        for c in CodeVariableClassifier.objects.exclude(name__regex="_old").order_by("name"):

            print("\n{}\n".format(c.name))
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

            dfs["turk"] = h.test_x
            dfs["turk"]['code_id'] = h.test_y
            dfs["turk"]['code'] = dfs["turk"]['code_id'].map(code_map)
            dfs['turk'].index = dfs['turk']['document_id']

            dfs["expert"] = h._get_expert_consensus_dataframe("1", coders=["pvankessel", "ahughes"], use_consensus_ignore_flag=(not options["original_consensus_subset_only"]))
            dfs['expert'].index = dfs['expert']['document_id']

            base_row['num_docs'] = len(dfs['expert'])

            for dfname, df in dfs.items():
                if is_not_null(df):
                    if "press_release" in c.document_types:
                        dfs[dfname]['party'] = dfs[dfname].apply(
                            lambda x: Document.objects.get(pk=x['document_id']).press_release.politician.party,
                            axis=1)
                    elif "facebook_post" in c.document_types:
                        dfs[dfname]['party'] = dfs[dfname].apply(lambda x: Document.objects.get(
                            pk=x['document_id']).facebook_post.source_page.politician.party, axis=1)
                        # dfs[dfname]['is_official'] = dfs[dfname].apply(
                        #     lambda x: Document.objects.get(
                        #         pk=x['document_id']).facebook_post.source_page.is_official,
                        #     axis=1)
                        # dfs[dfname] = dfs[dfname][dfs[dfname]['is_official'] == True]
                    dfs[dfname][dfs[dfname]['party'].isnull()]['party'] = "D"
                    dfs[dfname]["party"] = dfs[dfname]["party"].map(lambda x: str(x)[0])

            for pairname, df1, df2 in [
                ("expert_expert", "patrick", "adam"),
                ("turk_expert", "expert", "turk")
            ]:
                index_intersect = dfs[df1].index.intersection(dfs[df2].index)
                y1 = dfs[df1]['code'].ix[index_intersect].values
                y2 = dfs[df2]['code'].ix[index_intersect].values
                if sum(y1) > 0 and sum(y2) > 0:

                    result_dict = {0: defaultdict(int), 1: defaultdict(int)}
                    for a, b in zip(y1, y2):
                        result_dict[a][b] += 1
                    kappa = cohens_kappa([
                        [result_dict[0][0], result_dict[0][1]],
                        [result_dict[1][0], result_dict[1][1]]
                    ])
                    print("{} kappa: {} ({})".format(pairname, kappa["kappa"], kappa["std_kappa"]))
                    base_row["{}_kappa".format(pairname)] = kappa["kappa"]
                    base_row["{}_kappa_err".format(pairname)] = kappa["std_kappa"]

                    base_row["{}_pct_agree".format(pairname)] = numpy.average([1 if pair[0]==pair[1] else 0 for pair in zip(y1, y2)])

            # print base_row

            # Filter out validation cases that were also included in the training sample
            h.predict_y = pandas.Series(h.predict_y, index=h.test_x.index)
            h.test_x = h.test_x[~h.test_x['document_id'].isin(h.train_x['document_id'])]
            h.test_y = h.test_y.ix[h.test_x.index]
            h.predict_y = h.predict_y[h.test_x.index].values

            scores = c.get_code_cv_training_scores()
            if scores:
                row = copy.copy(base_row)
                row["partition"] = "all"
                row["scope"] = "cross_val_folds"
                row.update(scores)
                rows.append(row)
                # print "\nCV Training Scores"
                # print "Predicted mean: {} ({})".format(scores['pred_mean_mean'], scores['pred_mean_err'])
                # print "Actual mean: {} ({})".format(scores['true_mean_mean'], scores['true_mean_err'])
                # print "Precision: {} ({})".format(scores['precision_mean'], scores['precision_err'])
                # print "Recall: {} ({})".format(scores['recall_mean'], scores['recall_err'])
                print("model_cv_kappa: {} ({})".format(scores['kappa_mean'], scores['kappa_err']))
                # print "t-Test pass rate: {}".format(scores['ttest_pass_mean'])
                # print "\n"

            scores = c.get_code_validation_test_scores(use_expert_consensus_subset=True)
            if scores:
                row = copy.copy(base_row)
                row["partition"] = "all"
                row["scope"] = "holdout"
                row.update(scores)
                rows.append(row)
                # print "\nExpert Consensus Holdout Test Scores - Turk"
                # print "Predicted mean: {} ({})".format(scores['pred_mean'], scores['pred_err'])
                # print "Actual mean: {} ({})".format(scores['true_mean'], scores['true_err'])
                # print "Precision: {}".format(scores['precision'])
                # print "Recall: {}".format(scores['recall'])
                print("model_turk_kappa: {} ({})".format(scores['kappa'], scores['kappa_err']))
                # print "t-Test pass rate: {}".format(scores['ttest_pass'])
                # print "\n"

            scores = c.get_code_validation_test_scores(use_expert_consensus_subset=True, compute_for_experts=True)
            if scores:
                row = copy.copy(base_row)
                row["partition"] = "all"
                row["scope"] = "holdout_expert_consensus"
                row.update(scores)
                rows.append(row)
                # print "\nExpert Consensus Holdout Test Scores - Experts"
                # print "Predicted mean: {} ({})".format(scores['pred_mean'], scores['pred_err'])
                # print "Actual mean: {} ({})".format(scores['true_mean'], scores['true_err'])
                # print "Precision: {}".format(scores['precision'])
                # print "Recall: {}".format(scores['recall'])
                print("model_expert_kappa: {} ({})".format(scores['kappa'], scores['kappa_err']))
                # print "t-Test pass rate: {}".format(scores['ttest_pass'])
                # print "\n"

            for doc_type in c.document_types:
                if doc_type == "press_release":
                    partition = "press_release__politician__party"
                else:
                    partition = "facebook_post__source_page__politician__party"
                partition_scores = c.get_code_cv_training_scores(partition_by=partition,
                                                                 restrict_document_type=doc_type, min_support=50)
                if partition_scores:
                    for part, scores in partition_scores.items():
                        row = copy.copy(base_row)
                        row["partition"] = "{}_{}".format(doc_type, part)
                        row["scope"] = "cross_val_folds"
                        row.update(scores)
                        rows.append(row)
                        # print "\nCV Training Scores ({}, partition {})".format(doc_type, part)
                        # print "Predicted mean: {} ({})".format(scores['pred_mean_mean'], scores['pred_mean_err'])
                        # print "Actual mean: {} ({})".format(scores['true_mean_mean'], scores['true_mean_err'])
                        # print "Precision: {} ({})".format(scores['precision_mean'], scores['precision_err'])
                        # print "Recall: {} ({})".format(scores['recall_mean'], scores['recall_err'])
                        print("model_cv_kappa, {}, {}: {} ({})".format(doc_type, part, scores['kappa_mean'], scores['kappa_err']))
                        # print "t-Test pass rate: {}".format(scores['ttest_pass_mean'])
                        # print "\n"

                        #             for doc_type in c.document_types:
                        #                 if doc_type == "press_release":
                        #                     partition = "press_release__politician__party"
                        #                 else:
                        #                     partition = "facebook_post__source_page__politician__party"
                        #                 partition_scores = c.get_code_validation_test_scores(partition_by=partition, restrict_document_type=doc_type, use_expert_consensus_subset=True, min_support=50)
                        #                 if partition_scores:
                        #                     for part, scores in partition_scores.iteritems():
                        #                         row = copy.copy(base_row)
                        #                         row["partition"] = "{}_{}".format(doc_type, part)
                        #                         row["scope"] = "holdout"
                        #                         row.update(scores)
                        #                         rows.append(row)
                        #                         print "\nExpert Consensus Holdout Test Scores - Turk ({}, partition {})".format(doc_type, part)
                        #                         print "Predicted mean: {} ({})".format(scores['pred_mean'], scores['pred_err'])
                        #                         print "Actual mean: {} ({})".format(scores['true_mean'], scores['true_err'])
                        #                         print "Precision: {}".format(scores['precision'])
                        #                         print "Recall: {}".format(scores['recall'])
                        #                         print "Kappa: {} ({})".format(scores['kappa'], scores['kappa_err'])
                        #                         print "t-Test pass rate: {}".format(scores['ttest_pass'])
                        #                         print "\n"

                        #             for doc_type in c.document_types:
                        #                 if doc_type == "press_release":
                        #                     partition = "press_release__politician__party"
                        #                 else:
                        #                     partition = "facebook_post__source_page__politician__party"
                        #                 partition_scores = c.get_code_validation_test_scores(partition_by=partition, restrict_document_type=doc_type, min_support=50, use_expert_consensus_subset=True, compute_for_experts=True)
                        #                 if partition_scores:
                        #                     for part, scores in partition_scores.iteritems():
                        #                         row = copy.copy(base_row)
                        #                         row["partition"] = "{}_{}".format(doc_type, part)
                        #                         row["scope"] = "holdout_expert_consensus"
                        #                         row.update(scores)
                        #                         rows.append(row)
                        #                         print "\nExpert Consensus Holdout Test Scores - Experts ({}, partition {})".format(doc_type, part)
                        #                         print "Predicted mean: {} ({})".format(scores['pred_mean'], scores['pred_err'])
                        #                         print "Actual mean: {} ({})".format(scores['true_mean'], scores['true_err'])
                        #                         print "Precision: {}".format(scores['precision'])
                        #                         print "Recall: {}".format(scores['recall'])
                        #                         print "Kappa: {} ({})".format(scores['kappa'], scores['kappa_err'])
                        #                         print "t-Test pass rate: {}".format(scores['ttest_pass'])
                        #                         print "\n"

            finished.append(c.name)

        scores = pandas.DataFrame(rows)

        scores.ix[scores['partition'] == 'press_release_26.0', 'partition'] = "D"
        scores.ix[scores['partition'] == 'press_release_83.0', 'partition'] = "R"
        scores.ix[scores['partition'] == 'facebook_post_26', 'partition'] = "D"
        scores.ix[scores['partition'] == 'facebook_post_83', 'partition'] = "R"
        scores.ix[scores['partition'] == 'facebook_post_26.0', 'partition'] = "D"
        scores.ix[scores['partition'] == 'facebook_post_83.0', 'partition'] = "R"

        scores[(scores["partition"] == "all") & (scores["scope"] == "cross_val_folds")]

        rows = []
        for classifier, group in scores.groupby("classifier"):

            group_all = group[group["partition"] == "all"]
            row = {
                "classifier": classifier,
                "expert_holdout_docs": group_all["num_docs"].mean(),
                "turk_expert_kappa": group_all["turk_expert_kappa"].mean(),
                "turk_expert_kappa_err": group_all["turk_expert_kappa_err"].mean(),
                "turk_expert_pct_agree": group_all["turk_expert_pct_agree"].mean(),
                "expert_expert_kappa": group_all["expert_expert_kappa"].mean(),
                "expert_expert_kappa_err": group_all["expert_expert_kappa_err"].mean(),
                "expert_expert_pct_agree": group_all["expert_expert_pct_agree"].mean(),
                "model_turk_kappa": group_all[group_all["scope"] == "holdout"]["kappa"].mean(),
                "model_turk_kappa_err": group_all[group_all["scope"] == "holdout"]["kappa_err"].mean(),
                "model_turk_pct_agree": group_all[group_all["scope"] == "holdout"]["pct_agree"].mean(),
                "model_expert_kappa": group_all[group_all["scope"] == "holdout_expert_consensus"]["kappa"].mean(),
                "model_expert_kappa_err": group_all[group_all["scope"] == "holdout_expert_consensus"]["kappa_err"].mean(),
                "model_expert_pct_agree": group_all[group_all["scope"] == "holdout_expert_consensus"]["pct_agree"].mean()
            }

            for partition in ["all", "D", "R"]:
                cv_folds = group[(group["scope"] == "cross_val_folds") & (group["partition"] == partition)]
                row.update({
                    "cv_kappa_mean_{}".format(partition): cv_folds["kappa_mean"].mean(),
                    "cv_kappa_err_{}".format(partition): cv_folds["kappa_err"].mean(),
                    "cv_precision_mean_{}".format(partition): cv_folds["precision_mean"].mean(),
                    "cv_precision_err_{}".format(partition): cv_folds["precision_err"].mean(),
                    "cv_recall_mean_{}".format(partition): cv_folds["recall_mean"].mean(),
                    "cv_recall_err_{}".format(partition): cv_folds["recall_err"].mean(),
                    "cv_accuracy_mean_{}".format(partition): cv_folds["accuracy_mean"].mean(),
                    "cv_accuracy_err_{}".format(partition): cv_folds["accuracy_err"].mean(),
                    "cv_pred_mean_mean_{}".format(partition): cv_folds["pred_mean_mean"].mean(),
                    "cv_pred_mean_err_{}".format(partition): cv_folds["pred_mean_err"].mean(),
                    "cv_true_mean_mean_{}".format(partition): cv_folds["true_mean_mean"].mean(),
                    "cv_true_mean_err_{}".format(partition): cv_folds["true_mean_err"].mean(),
                    "cv_pct_agree_mean_{}".format(partition): cv_folds["pct_agree_mean"].mean(),
                    "cv_pct_agree_err_{}".format(partition): cv_folds["pct_agree_err"].mean()
                })

            rows.append(row)

        scores_output_df = pandas.DataFrame(rows)

        ordered_cols = [
            "classifier",
            "expert_holdout_docs",

            "expert_expert_kappa", "expert_expert_kappa_err", "expert_expert_pct_agree",
            "turk_expert_kappa", "turk_expert_kappa_err", "turk_expert_pct_agree",
            "model_expert_kappa", "model_expert_kappa_err", "model_expert_pct_agree",
            "model_turk_kappa", "model_turk_kappa_err", "model_turk_pct_agree",

            "cv_kappa_mean_all", "cv_kappa_err_all",
            "cv_kappa_mean_D", "cv_kappa_err_D",
            "cv_kappa_mean_R", "cv_kappa_err_R",

            "cv_accuracy_mean_all", "cv_accuracy_err_all",
            "cv_accuracy_mean_D", "cv_accuracy_err_D",
            "cv_accuracy_mean_R", "cv_accuracy_err_R",

            "cv_precision_mean_all", "cv_precision_err_all",
            "cv_precision_mean_D", "cv_precision_err_D",
            "cv_precision_mean_R", "cv_precision_err_R",

            "cv_recall_mean_all", "cv_recall_err_all",
            "cv_recall_mean_D", "cv_recall_err_D",
            "cv_recall_mean_R", "cv_recall_err_R",

            "cv_pred_mean_mean_all", "cv_pred_mean_err_all",
            "cv_true_mean_mean_all", "cv_true_mean_err_all",
            "cv_pred_mean_mean_D", "cv_pred_mean_err_D",
            "cv_true_mean_mean_D", "cv_true_mean_err_D",
            "cv_pred_mean_mean_R", "cv_pred_mean_err_R",
            "cv_true_mean_mean_R", "cv_true_mean_err_R",

            "cv_pct_agree_mean_all", "cv_pct_agree_err_all",
            "cv_pct_agree_mean_D", "cv_pct_agree_err_D",
            "cv_pct_agree_mean_R", "cv_pct_agree_err_R"
        ]

        scores_output_df[[c for c in ordered_cols if "_err" not in c]]

        # scores_output_df[ordered_cols].to_csv("model_scores_final.csv")
        if options["original_consensus_subset_only"]:
            suffix = "_consensus_subset"
        else:
            suffix = ""

        h = FileHandler("output/queries/partisan_antipathy",
            use_s3=True,
            bucket=settings.S3_BUCKET,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY
        )
        h.write("model_scores_final{}".format(suffix), scores_output_df[ordered_cols], format="csv")