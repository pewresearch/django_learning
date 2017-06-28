import pandas, math, re, numpy, cPickle, copy

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

        parser.add_argument("var_name", type=str)
        parser.add_argument("--top_n", default=100, type=int)
        parser.add_argument("--max_length", default=10000, type=int)
        parser.add_argument("--min_likes", default=0, type=int)
        parser.add_argument("--use_raw", default=False, action="store_true")
        parser.add_argument("--no_party_grouping", default=False, action="store_true")
        parser.add_argument("--group_by_week", default=False, action="store_true")

    def handle(self, *args, **options):

        groupby = []

        if not options["no_party_grouping"]: groupby.append("party")
        if options["group_by_week"]: groupby.append("week")


        c = CodeVariable.objects.get(name=options["var_name"])

        frame_docs = ClassifierDocumentCode.objects \
            .filter(code__variable=c) \
            .filter(document__facebook_post__isnull=False) \
            .filter(document__facebook_post__source_page__is_official=True)

        docs = ClassifierDocumentCode.objects \
            .filter(code__variable=c)\
            .filter(code__value="1")\
            .filter(document__facebook_post__isnull=False) \
            .filter(document__facebook_post__source_page__is_official=True)

        if options["var_name"] == "expresses_anger":

            print "NOTE: filtering down to anger AND opposition (we hard-coded it this way for the report)"
            opposition_ids = []
            for var in ["oppose_obama", "oppose_dems", "oppose_reps"]:
                opposition_ids.extend(
                    ClassifierDocumentCode.objects\
                        .filter(code__variable=CodeVariable.objects.get(name=var))\
                        .filter(code__value="1")\
                        .filter(document__facebook_post__isnull=False)\
                        .filter(document__facebook_post__source_page__is_official=True)\
                        .values_list("document_id", flat=True)
                )
            docs = docs.filter(document_id__in=opposition_ids)

        elif options["var_name"] == "bipartisanship":

            print "NOTE: filtering out anger and opposition (we hard-coded it this way for the report)"
            opposition_ids = []
            for var in ["oppose_obama", "oppose_dems", "oppose_reps", "expresses_anger"]:
                opposition_ids.extend(
                    ClassifierDocumentCode.objects \
                        .filter(code__variable=CodeVariable.objects.get(name=var)) \
                        .filter(code__value="1") \
                        .filter(document__facebook_post__isnull=False) \
                        .filter(document__facebook_post__source_page__is_official=True) \
                        .values_list("document_id", flat=True)
                )
            docs = docs.exclude(document_id__in=opposition_ids)

        frame_df = pandas.DataFrame.from_records(
            frame_docs.values(
                "document__facebook_post__source_page__politician__bioguide_id",
                "document__facebook_post__likes",
                "document__facebook_post__total_comments",
                "document__facebook_post__shares"
            ).distinct()
        )

        df = pandas.DataFrame.from_records(
            docs.values(
                "document_id",
                "document__facebook_post__source_page__politician__bioguide_id",
                "document__facebook_post__source_page__politician__last_name",
                "document__facebook_post__source_page__politician__party__name",
                "document__facebook_post__likes",
                "document__facebook_post__total_comments",
                "document__facebook_post__shares",
                "document__text",
                "document__date",
                "document__facebook_post__facebook_id",
                "document__facebook_post__source_page__facebook_id",
                "code__value"
            ).distinct()
        )

        frame_df = frame_df.rename(columns={
            'document__facebook_post__likes': 'fb_post_likes',
            'document__facebook_post__total_comments': 'fb_post_total_comments',
            'document__facebook_post__source_page__politician__bioguide_id': 'bioguide_id',
            "document__facebook_post__shares": 'fb_post_shares'
        })

        df = df.rename(columns={
            'document__facebook_post__source_page__politician__last_name': 'last_name',
            'document__facebook_post__likes': 'fb_post_likes',
            'document__text': 'text',
            'document__facebook_post__source_page__politician__bioguide_id': 'bioguide_id',
            'document__facebook_post__total_comments': 'fb_post_total_comments',
            'document__facebook_post__shares': 'fb_post_shares',
            'document__facebook_post__source_page__politician__party__name': 'party',
            'code__value': options["var_name"],
            "document__facebook_post__facebook_id": "fbid",
            "document__facebook_post__source_page__facebook_id": "page_fbid"
        })

        df[df['party'].isnull()]['party'] = "Democratic Party"

        df["post_fbid"] = df["fbid"].map(lambda x: x.split("_")[-1])

        df["fb_post_likes_pol_avg"] = df['bioguide_id'].map(
            lambda x: frame_df[frame_df['bioguide_id']==x]['fb_post_likes'].mean())
        df["fb_post_total_comments_pol_avg"] = df['bioguide_id'].map(
            lambda x: frame_df[frame_df['bioguide_id'] == x]['fb_post_total_comments'].mean())
        df["fb_post_shares_pol_avg"] = df['bioguide_id'].map(
            lambda x: frame_df[frame_df['bioguide_id'] == x]['fb_post_shares'].mean())

        df["fb_post_likes_pol_std"] = df['bioguide_id'].map(
            lambda x: frame_df[frame_df['bioguide_id'] == x]['fb_post_likes'].std())
        df["fb_post_total_comments_pol_std"] = df['bioguide_id'].map(
            lambda x: frame_df[frame_df['bioguide_id'] == x]['fb_post_total_comments'].std())
        df["fb_post_shares_pol_std"] = df['bioguide_id'].map(
            lambda x: frame_df[frame_df['bioguide_id'] == x]['fb_post_shares'].std())

        df['fb_post_likes_norm'] = df.apply(
            lambda x: (x['fb_post_likes'] - x["fb_post_likes_pol_avg"]) / x["fb_post_likes_pol_std"], axis=1)
        df['fb_post_total_comments_norm'] = df.apply(
            lambda x: (x['fb_post_total_comments'] - x["fb_post_total_comments_pol_avg"]) / x["fb_post_total_comments_pol_std"], axis=1)
        df['fb_post_shares_norm'] = df.apply(
            lambda x: (x['fb_post_shares'] - x["fb_post_shares_pol_avg"]) / x[
                "fb_post_shares_pol_std"], axis=1)

        # zscore = lambda x: (x - x.mean()) / x.std()
        # pol_avg = lambda x: x.mean()
        # df["fb_post_likes_norm"] = df.groupby("bioguide_id")["fb_post_likes"].transform(zscore)
        # df["fb_post_likes_pol_avg"] = df.groupby("bioguide_id")["fb_post_likes"].transform(pol_avg)

        df["fb_post_like_multiplier"] = df["fb_post_likes"] / df["fb_post_likes_pol_avg"]
        df["fb_post_total_comments_multiplier"] = df["fb_post_total_comments"] / df["fb_post_total_comments_pol_avg"]
        df["fb_post_share_multiplier"] = df["fb_post_shares"] / df["fb_post_shares_pol_avg"]

        df["date"] = df["document__date"].dt.date
        df['week'] = df["document__date"].dt.week
        df['month'] = df["document__date"].dt.month
        df['length'] = df['text'].map(lambda x: len(x))
        df['party'] = df['party']
        del df['document__date']

        df['url'] = df.apply(lambda x: "http://www.facebook.com/{}/posts/{}".format(x['page_fbid'], x['post_fbid']), axis=1)

        rows = []
        for date, group in df[(df['length'] <= options["max_length"]) & (df['fb_post_likes'] >= options["min_likes"])].groupby(groupby):
            # rows.extend(group.sort("document__facebook_post__likes", ascending=False).head(top_n).to_dict("records"))
            rows.extend(
                group.sort("fb_post_likes_norm" if not options["use_raw"] else "fb_post_likes", ascending=False).head(options["top_n"]).to_dict(
                    "records"))

        df_top = pandas.DataFrame(rows)

        h = FileHandler("output/queries/partisan_antipathy",
            use_s3=True,
            bucket=settings.S3_BUCKET,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY
        )
        h.write("top_{}_{}_posts_by_{}".format(
            options["top_n"],
            options["var_name"],
            "_".join(groupby)
        ), df_top, format="csv")