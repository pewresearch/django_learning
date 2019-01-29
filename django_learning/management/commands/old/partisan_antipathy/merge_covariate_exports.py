from __future__ import print_function
import pandas, datetime

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from logos.models import *
from logos.utils import *


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        pass

    def handle(self, *args, **options):

        from logos.utils.io import FileHandler
        h = FileHandler("output/queries/partisan_antipathy",
            use_s3=True,
            bucket=settings.S3_BUCKET,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY
        )

        print("Processing Facebook Posts")

        fb_posts = h.read("facebook_post_stats", format="csv")
        # fb_posts = FacebookPost.objects.filter(
        #     document__in=DocumentSampleFrame.objects.get(name="facebook_posts_jan15_apr16").documents.all())
        # fb_posts = pandas.DataFrame.from_records(fb_posts.values())
        fb_posts.columns = ["fb_post_{}".format(c) for c in fb_posts.columns]

        # NOTE: the facebook_post_stats were computed for June 2016, when they were resynced; however, shares were last synced in November 2016
        # accordingly, we'll pull in the latest data here
        fb_post_shares = FacebookPost.objects.filter(
            document__in=DocumentSampleFrame.objects.get(name="facebook_posts_jan15_apr16").documents.all())
        fb_post_shares = pandas.DataFrame.from_records(fb_post_shares.values("id", "shares"))
        fb_posts = fb_posts.merge(fb_post_shares, how="left", left_on="fb_post_id", right_on="id")
        del fb_posts['fb_post_shares']
        fb_posts = fb_posts.rename(columns={"shares": "fb_post_shares"})

        fb_post_documents = DocumentSampleFrame.objects.get(name="facebook_posts_jan15_apr16").documents.values("id", "facebook_post_id", "facebook_post__source_page_id", "date")
        fb_post_documents = pandas.DataFrame.from_records(fb_post_documents)
        fb_post_documents = fb_post_documents.rename(columns={"id": "document_id", "facebook_post__source_page_id": "fb_post_source_page_id"})
        fb_posts = fb_posts.merge(fb_post_documents, how="left", left_on="fb_post_id", right_on="facebook_post_id")

        fb_post_codes = h.read("facebook_post_document_codes", format="csv")
        fb_post_codes.columns = ["fb_post_{}".format(c) for c in fb_post_codes.columns]
        fb_page_stats = h.read("facebook_page_stats", format="csv")
        fb_page_stats.columns = ["fb_page_{}".format(c) for c in fb_page_stats.columns]

        fb_posts_merged = fb_posts.merge(fb_page_stats, how='left', left_on='fb_post_source_page_id',
                                         right_on='fb_page_id')
        for delcol in [
            "fb_post_Unnamed: 0", "fb_post_caption", "fb_post_comment_backfill", "fb_post_description", "fb_post_last_update_time",
            "fb_post_link", "fb_post_message", "fb_post_source", "fb_post_story", "fb_post_title",
            "fb_post_year_id", "fb_page_Unnamed: 0", "fb_page_about", "fb_page_bio", "fb_page_birthday",
            "fb_page_city", "fb_page_country", "fb_page_event_backfill", "fb_page_hometown", "fb_page_state_id",
            "fb_page_last_update_time", "fb_page_name", "fb_page_street", "fb_page_websites", "fb_page_zip",
            "fb_page_username", "fb_page_history_id", "fb_page_history_type", "fb_page_history_user_id", "fb_page_is_verified",
            "fb_page_subcategories", "fb_page_feed_backfill", "fb_page_category"
        ]:
            if delcol in fb_posts_merged.columns:
                del fb_posts_merged[delcol]

        pols = h.read("politician_basic_info", format="csv")
        fb_posts_merged = fb_posts_merged.merge(pols[["id", "bioguide_id"]], how="left",
                                                left_on="fb_page_politician_id", right_on="id")

        fb_posts_merged = fb_posts_merged.merge(fb_post_codes, how="left", left_on="document_id",
                                                right_on="fb_post_document_id")
        h.write("facebook_posts_merged", fb_posts_merged, format="csv")

        print("Processing Press Releases")

        prs = PressRelease.objects.filter(
            document__in=DocumentSampleFrame.objects.get(name="press_releases_jan15_apr16_all_sources").documents.all())
        prs = pandas.DataFrame.from_records(prs.values())
        prs.columns = ["pr_{}".format(c) for c in prs.columns]

        pr_documents = DocumentSampleFrame.objects.get(name="press_releases_jan15_apr16_all_sources").documents.values(
            "id", "press_release_id", "date")
        pr_documents = pandas.DataFrame.from_records(pr_documents)

        pr_documents = pr_documents.rename(columns={"id": "document_id"})
        prs = prs.merge(pr_documents, how="left", left_on="pr_id", right_on="press_release_id")

        pr_codes = h.read("press_release_document_codes", format="csv")
        pr_codes.columns = ["pr_{}".format(c) for c in pr_codes.columns]

        prs_merged = prs.merge(pr_codes, how="left", left_on="document_id", right_on="pr_document_id")
        for delcol in [
            "pr_content_dates",
            "pr_content_links",
            "pr_duplicate_ids",
            "pr_duplicate_legacy_ids",
            "pr_duplicate_source_ids",
            "pr_first_parse_time",
            "pr_guessed_date",
            "pr_latest_parse_time",
            "pr_legacy_id",
            "pr_original_list_page_date",
            "pr_original_date",
            "pr_source_id",
            "pr_title"
        ]:
            del prs_merged[delcol]

        pols = h.read("politician_basic_info", format="csv")
        prs_merged = prs_merged.merge(pols[["id", "bioguide_id"]], how="left", left_on="pr_politician_id",
                                      right_on="id")

        # prs_merged.to_csv("../final_data/press_releases_merged.csv")
        h.write("press_releases_merged", prs_merged, format="csv")

        print("Processing Politicians")

        pols = h.read("politician_basic_info", format="csv")
        pols_campaign_stats = h.read("politician_campaign_stats", format="csv")
        pols_district_demographics = h.read("politician_district_demographics", format="csv")
        pols_election_info = h.read("politician_election_info", format="csv")
        pols_legislative_stats = h.read("politician_legislative_stats", format="csv")
        pols_personal_metrics = h.read("politician_personal_metrics", format="csv")
        pols_vote_categories = h.read("politician_vote_categories", format="csv")

        pols = pols.merge(pols_campaign_stats, left_on="id", right_on="id", suffixes=("", "_extra"), how="left")

        pols = pols.merge(pols_vote_categories, left_on="id", right_on="id", suffixes=("", "_extra"), how="left")

        pols_election_info = pols_election_info[
            ["id", "elections", "most_recent_election_year", "most_recent_primary_year",
             "most_recent_election_percent", "most_recent_election_win_margin", "most_recent_election_won",
             "most_recent_primary_percent", "most_recent_primary_win_margin", "most_recent_primary_won"]]

        pols = pols.merge(pols_election_info, left_on="id", right_on="id", suffixes=("", "_extra"), how="left")

        pols_legislative_stats = pols_legislative_stats[
            ["id", "avg_hearings_per_cosponsored_bill", "avg_hearings_per_sponsored_bill",
             "avg_records_per_cosponsored_bill",
             "avg_records_per_sponsored_bill", "bills_cosponsored", "bills_cosponsored_avg_cosponsors",
             "bills_cosponsored_passed",
             "bills_sponsored", "bills_sponsored_avg_cosponsors", "bills_sponsored_passed_pct", "hearings_attended",
             "hearings_attended_avg_witnesses", "hearings_held", "in_Congressional Progressive Caucus", "in_Freedom Caucus",
             "in_Liberty Caucus", "in_Tea Party Caucus", "num_caucuses",
             "num_congress_committees_total", "num_congress_subcommittees_total",
             "num_congress_committees_current", "num_congress_subcommittees_current",
             "num_congress_committees_chaired_current", "num_congress_subcommittees_chaired_current",
             "num_congress_committees_ranking_member_current", "num_congress_subcommittees_ranking_member_current",
             "num_congress_committees_chaired_all_time", "num_congress_subcommittees_chaired_all_time",
             "num_congress_committees_ranking_member_all_time", "num_congress_subcommittees_ranking_member_all_time",
             "total_hearings_for_cosponsored_bills", "total_hearings_for_sponsored_bills",
             "total_records_for_cosponsored_bills", "total_records_for_sponsored_bills", "votes_abstained", "votes_against",
             "votes_against_bill_passed", "votes_for", "votes_for_bill_passed"]]
        pols = pols.merge(pols_legislative_stats, left_on="id", right_on="id", suffixes=("", "_extra"), how="left")

        pols_personal_metrics = pols_personal_metrics[[
            "id",
            u'metrics_latest_tea_party_elect',
            u'metrics_latest_tea_party_elect_year',
            u'metrics_latest_tea_party_endorse',
            u'metrics_latest_tea_party_endorse_year',
            "metrics_latest_dw_nominate1",
            "metrics_latest_dw_nominate2",
            "metrics_latest_dw_nominate1_year",
            "metrics_latest_dw_nominate2_year"
        ]]
        pols = pols.merge(pols_personal_metrics, left_on="id", right_on="id", suffixes=("", "_extra"), how="left")

        pols_district_demographics = pols_district_demographics[[
            "id", "district",
            "district_age_18_and_over_pct_of_total_population_2013",
            "district_age_median_2013",
            "district_civilian_population_pct_of_total_population_2013",
            "district_educ_bachelors_degree_pct_of_age_25_and_over_2013",
            "district_educ_graduate_degree_pct_of_age_25_and_over_2013",
            "district_eth_hisp_pct_of_total_population_2013",
            "district_health_insurance_not_covered_pct_of_total_population_2013",
            "district_housing_occupied_units_owner_median_value_2013",
            "district_housing_occupied_units_renter_median_rent_2013",
            "district_housing_occupied_units_owner_pct_of_housing_occupied_units_2013",
            "district_housing_occupied_units_renter_pct_of_housing_occupied_units_2013",
            "district_income_household_mean_2013",
            "district_income_household_median_2013",
            "district_jobs_in_labor_force_pct_of_age_16_and_over_2013",
            "district_origin_native_born_outside_us_pct_of_total_population_2013",
            "district_origin_native_born_us_pct_of_total_population_2013",
            "district_poverty_households_pct_2013",
            "district_race_nonwhite_pct_of_total_population_2013",
            "district_sex_female_pct_of_total_population_2013",
            "district_total_population_2013",
            "district_unemployment_rate_2013",
            "district_pvi_2014",
            "district_pvi_2013",
            "district_vote_share_mccain_08",
            "district_vote_share_obama_08",
            "district_vote_share_obama_12",
            "district_vote_share_romney_12"
        ]]
        pols = pols.merge(pols_district_demographics, left_on="id", right_on="id", suffixes=("", "_extra"), how="left")

        pols = pols[[c for c in pols.columns if not c.endswith("_extra")]]

        # pols.to_csv("../final_data/politicians_merged.csv")
        h.write("politicians_merged", pols, format="csv")

        # print "Compiling aggregate dataset"
        #
        # fb_post_pols = fb_posts_merged.groupby("fb_page_politician_id").mean().reset_index()
        # for delcol in ["fb_post_id", "fb_post_source_page_id", "facebook_post_id", "document_id", "fb_page_facebook_id",
        #                "fb_post_document_id"]:
        #     del fb_post_pols[delcol]
        # pols_with_codes = pols.merge(fb_post_pols, left_on="id", right_on="fb_page_politician_id")
        # del pols_with_codes['id_y']
        # pols_with_codes = pols_with_codes.rename(columns={"id_x": "id"})
        #
        # pr_pols = prs_merged.groupby("pr_politician_id").mean().reset_index()
        # for delcol in ["document_id", "pr_year_id", "pr_id", "press_release_id", "pr_document_id"]:
        #     del pr_pols[delcol]
        # pols_with_codes = pols_with_codes.merge(pr_pols, left_on="id", right_on="pr_politician_id")
        # del pols_with_codes['id_y']
        # pols_with_codes = pols_with_codes.rename(columns={"id_x": "id"})
        # # pols_with_codes.to_csv("../final_data/politicians_merged_with_codes.csv")
        # h.write("politicians_merged_with_codes", pols_with_codes, format="csv")
        #
        # fb_post_pols = fb_posts_merged[fb_posts_merged['fb_page_is_official']].groupby(
        #     "fb_page_politician_id").mean().reset_index()
        # for delcol in ["fb_post_id", "fb_post_source_page_id", "facebook_post_id", "document_id", "fb_page_facebook_id",
        #                "fb_post_document_id"]:
        #     del fb_post_pols[delcol]
        # pols_with_codes = pols.merge(fb_post_pols, left_on="id", right_on="fb_page_politician_id")
        # del pols_with_codes['id_y']
        # pols_with_codes = pols_with_codes.rename(columns={"id_x": "id"})
        #
        # pr_pols = prs_merged.groupby("pr_politician_id").mean().reset_index()
        # for delcol in ["document_id", "pr_year_id", "pr_id", "press_release_id", "pr_document_id"]:
        #     del pr_pols[delcol]
        # pols_with_codes = pols_with_codes.merge(pr_pols, left_on="id", right_on="pr_politician_id")
        # del pols_with_codes['id_y']
        # pols_with_codes = pols_with_codes.rename(columns={"id_x": "id"})
        # # pols_with_codes.to_csv("../final_data/politicians_merged_with_codes_official_only.csv")
        # h.write("politicians_merged_with_codes_official_only", pols_with_codes, format="csv")

