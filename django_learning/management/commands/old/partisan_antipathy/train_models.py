import pandas, itertools, numpy

from contextlib import closing

from django.core.management.base import BaseCommand, CommandError
from django.db import models

from logos.learning.supervised import DocumentClassificationHandler
from logos.utils.io import FileHandler
from logos.utils import is_null, is_not_null


class Command(BaseCommand):

    """
    """

    help = ""

    def add_arguments(self, parser):

        parser.add_argument("code_variable_name", type=str)
        parser.add_argument("--pipeline", default=None, type=str)
        parser.add_argument("--refresh_training_data", default=False, action="store_true")
        parser.add_argument("--refresh_model", default=False, action="store_true")
        parser.add_argument("--num_cores", default=2, type=int)
        parser.add_argument("--save_to_database", default=None, type=str)
        parser.add_argument("--document_type", default=None, type=str)
        parser.add_argument("--use_consensus_weights", default=False, action="store_true")
        parser.add_argument("--compute_cv_folds", default=False, action="store_true")

    def handle(self, *args, **options):

        code_variable_name = options.pop("code_variable_name")
        refresh_training_data = options.pop("refresh_training_data")
        refresh_model = options.pop("refresh_model")
        save_name = options.pop("save_to_database")
        use_consensus_weights = options.pop("use_consensus_weights")
        compute_cv_folds = options.pop("compute_cv_folds")
        document_type = options.pop("document_type")

        thresholds = {
            "bipartisanship": [.3],
            "district_benefit": [.5],
            "expresses_anger": [.3],
            "foreign_policy": [.3],
            "oppose_agency": [.3],
            "oppose_any": [.3],
            "oppose_courts": [.3],
            "oppose_dems": [.3],
            "oppose_dems_with_anger": [.3],
            "oppose_obama": [.3],
            "oppose_obama_with_anger": [.3],
            "oppose_org": [.3],
            "oppose_political_figures": [.3],
            "oppose_reps": [.3],
            "oppose_reps_with_anger": [.3],
            "undermines_trust": [.5, .7],
            "oppose_political_figures_with_anger": [.3]
        }

        if document_type: doc_types = [document_type]
        else: doc_types = ["facebook_post", "press_release"]
        for doc_type in doc_types:
            if doc_type == "press_release": frame = "press_releases_jan15_apr16_all_sources"
            else: frame = "facebook_posts_jan15_apr16"
            for t in thresholds[code_variable_name]:
                print "Computing {} for {} at threshold {}".format(code_variable_name, doc_type, t)
                params = {
                    "codes": {"mturk": {"consolidation_threshold": t}},
                    "documents": {"frames": [frame], "balance_document_types": False, "include_frame_weights": False},
                    "model": {"use_sample_weights": True}
                }
                if use_consensus_weights:
                    params["codes"]["use_consensus_weights"] = True
                h = DocumentClassificationHandler(
                    doc_type,
                    code_variable_name,
                    params=params,
                    **options
                )
                h.load_training_data(refresh=refresh_training_data)
                h.load_model(refresh=refresh_model)
                h.print_report()
                if save_name:
                    # h.save_to_database("{}_{}".format(save_name, t), compute_cv_folds=True)
                    h.save_to_database(
                        # "{}_{}_{}{}".format(save_name, doc_type, t, "_consensus_weighted" if use_consensus_weights else ""),
                        "{}_{}".format(save_name, doc_type),
                        compute_cv_folds=compute_cv_folds
                    )