import pandas, datetime
from django.core.management.base import BaseCommand, CommandError

from logos.models import *


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("filename", type=str)
        parser.add_argument("--refresh", default=False, action="store_true")

    def handle(self, *args, **options):
        
        min_date = datetime.date(2015, 1, 1)
        min_date_actual = datetime.date(2015, 1, 6)
        max_date = datetime.date(2016, 5, 1)
        as_of_date = datetime.datetime(2016, 11, 15)

        filename = options.pop("filename")

        if filename == "classifier_precision_recall_scores":

            rows = []
            from django.db.models import Count
            for c in CodeVariableClassifier.objects.annotate(c=Count("variable__codes")).filter(c=2):
                pos_scores = c.get_code_cv_training_scores(code_value="1")
                neg_scores = c.get_code_cv_training_scores(code_value="0")
                if pos_scores and neg_scores:
                    p1 = 1.0 - pos_scores['precision_mean']
                    r1 = 1.0 - pos_scores['recall_mean']
                    p0 = 1.0 - neg_scores['precision_mean']
                    r0 = 1.0 - neg_scores['recall_mean']
                    rows.append({
                        "name": c.name,
                        "p1": p1,
                        "p0": p0,
                        "r1": r1,
                        "r0": r0
                    })
            from logos.utils.io import FileHandler
            df = pandas.DataFrame(rows)
            h = FileHandler("output/queries/partisan_antipathy", use_s3=True)
            h.write("classifier_precision_recall_scores", df, format="csv")

        elif filename.startswith("politician_"):

            pols = Politician.objects.us_congress_members(during_year=2015)

            if filename == "politician_basic_info":
                pols.dataframe("politician_basic_info", export=True, export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "politician_district_demographics":
                pols.dataframe("politician_district_demographics", during_year=2015, export=True, export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "politician_personal_metrics":
                pols.dataframe("politician_personal_metrics", export=True, export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "politician_election_info":
                pols.dataframe("politician_election_info", export=True, export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "politician_campaign_stats":
                pols.dataframe("politician_campaign_stats", max_year=2015, export=True, export_folder="partisan_antipathy",
                               refresh=options["refresh"])
            elif filename == "politician_legislative_stats":
                pols.dataframe("politician_legislative_stats", session_num=114, export=True,
                               export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "politician_vote_categories":
                pols.dataframe("politician_vote_categories", start_date=min_date, end_date=max_date, export=True,
                               export_folder="partisan_antipathy", refresh=options["refresh"])

        elif filename.startswith("facebook_"):

            fb_frame = DocumentSampleFrame.objects.get(name="facebook_posts_jan15_apr16")
            posts = FacebookPost.objects.filter(pk__in=fb_frame.documents.values_list("facebook_post_id", flat=True))
            pages = FacebookPage.objects.filter(
                pk__in=fb_frame.documents.values_list("facebook_post__source_page_id", flat=True))

            if filename == "facebook_post_stats":
                posts.dataframe("facebook_post_stats", export=True, min_date=min_date, max_date=max_date, as_of_date=as_of_date,
                                export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "facebook_page_stats":
                pages.dataframe("facebook_page_stats", export=True, min_date=min_date, max_date=max_date, as_of_date=as_of_date,
                                export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "facebook_post_document_codes":
                fb_frame.documents.dataframe("document_codes", document_type="facebook_post", export=True,
                                             export_name="facebook_post_document_codes", export_folder="partisan_antipathy",
                                             refresh=options["refresh"])
            elif filename == "facebook_post_document_entities":
                fb_frame.documents.dataframe("document_entities", export=True,
                                             export_name="facebook_post_document_entities",
                                             export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "facebook_post_document_text":
                fb_frame.documents.dataframe("document_text", export=True, export_name="facebook_post_document_text",
                                             export_folder="partisan_antipathy", refresh=options["refresh"])

        elif filename.startswith("press_release_"):

            pr_frame = DocumentSampleFrame.objects.get(name="press_releases_jan15_apr16_all_sources")

            if filename == "press_release_document_codes":
                pr_frame.documents.dataframe("document_codes", document_type="press_release", export=True,
                                             export_name="press_release_document_codes",
                                             export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "press_release_document_entities":
                pr_frame.documents.dataframe("document_entities", export=True,
                                             export_name="press_release_document_entities",
                                             export_folder="partisan_antipathy", refresh=options["refresh"])
            elif filename == "press_release_document_text":
                pr_frame.documents.dataframe("document_text", export=True, export_name="press_release_document_text",
                                             export_folder="partisan_antipathy", refresh=options["refresh"])