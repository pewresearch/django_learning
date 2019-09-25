from __future__ import print_function

import pandas as pd

from django_pewtils import get_model
from pewtils import is_not_null
from django_learning.utils.dataset_document_filters import dataset_document_filters
from pewanalytics.stats.sampling import compute_sample_weights_from_frame


def extract_sample():
    pass

def extract_sampling_frame():
    pass

def get_sampling_weights(
    samples,
    refresh_flags=False,
    ignore_stratification_weights=False,
    document_filters=None
):

    frame_ids = set(list(samples.values_list("frame_id", flat=True)))
    if len(frame_ids) == 1:
        sampling_frame = get_model("SamplingFrame", app_name="django_learning").objects.get(pk__in=frame_ids)
    else:
        raise Exception("All of your samples must be belong to the same sampling frame")

    keyword_weight_columns = set()
    strat_vars = set()
    additional_vars = set()
    for sample in samples:
        params = sample.get_params()
        stratify_by = params.get("stratify_by", None)
        if is_not_null(stratify_by):
            strat_vars.add(stratify_by)
        sampling_searches = params.get("sampling_searches", {})
        if len(sampling_searches) > 0:
            for search_name in list(sampling_searches.keys()):
                keyword_weight_columns.add(search_name)
        for additional_var in list(params.get("additional_weights", {}).keys()):
            additional_vars.add(additional_var)

    frame = sampling_frame.get_sampling_flags(refresh=refresh_flags, sampling_search_subset=keyword_weight_columns)
    if len(keyword_weight_columns) > 0:
        keyword_weight_columns = set(["search_{}".format(s) for s in keyword_weight_columns])
        keyword_weight_columns.add("search_none")
    if document_filters:
        # print("Applying frame document filter: {}".format(len(frame)))
        frame = frame.rename(columns={"pk": "document_id"})
        for filter_name, filter_args, filter_kwargs in document_filters:
            frame = dataset_document_filters[filter_name](None, frame, *filter_args, **filter_kwargs)
        frame = frame.rename(columns={"document_id": "pk"})
        # print("Frame is now {}".format(len(frame)))
    # if filter_params:
    #     frame = frame[frame["pk"].isin(
    #         filter_queryset_by_params(sampling_frame.documents.all(), filter_params).values_list("pk", flat=True)
    #     )]

    if ignore_stratification_weights:
        strat_vars = set()

    strat_weight_columns = set()
    for stratify_by in strat_vars:
        dummies = pd.get_dummies(frame[stratify_by], prefix=stratify_by)
        strat_weight_columns = strat_weight_columns.union(set(dummies.columns))
        frame = frame.join(dummies)

    additional_weight_columns = set()
    for additional in additional_vars:
        dummies = pd.get_dummies(frame[additional], prefix=additional)
        additional_weight_columns = additional_weight_columns.union(set(dummies.columns))
        frame = frame.join(dummies)

    if "search_none" in keyword_weight_columns:
        actual_keywords = [k for k in list(keyword_weight_columns) if k != "search_none"]
        frame['search_none'] = ~frame[actual_keywords].max(axis=1)
    full_sample = frame[frame['pk'].isin(list(
        get_model("SampleUnit", app_name="django_learning").objects.filter(sample__in=samples).values_list(
            "document_id", flat=True)))].reset_index()

    if len(keyword_weight_columns) > 0:
        keyword_weight = compute_sample_weights_from_frame(frame, full_sample, list(keyword_weight_columns))
        full_sample['keyword_weight'] = keyword_weight
    else:
        full_sample['keyword_weight'] = None

    if len(strat_weight_columns) > 0:
        strat_weight = compute_sample_weights_from_frame(frame, full_sample, list(strat_weight_columns))
        full_sample['strat_weight'] = strat_weight
    else:
        full_sample['strat_weight'] = None

    if len(additional_weight_columns) > 0:
        additional_weight = compute_sample_weights_from_frame(frame, full_sample, list(additional_weight_columns))
        full_sample['additional_weight'] = additional_weight
    else:
        full_sample['additional_weight'] = None

    full_sample['approx_weight'] = 1.0
    if len(keyword_weight_columns) > 0:
        full_sample['approx_weight'] *= keyword_weight
    if len(strat_weight_columns) > 0:
        full_sample['approx_weight'] *= strat_weight
    if len(additional_weight_columns) > 0:
        full_sample['approx_weight'] *= additional_weight

    all_weight_columns = keyword_weight_columns.union(strat_weight_columns).union(additional_weight_columns)
    full_weight = compute_sample_weights_from_frame(frame, full_sample, list(all_weight_columns))
    full_sample['weight'] = full_weight
    full_sample['weight'] = full_sample['weight'].fillna(1.0)

    return full_sample