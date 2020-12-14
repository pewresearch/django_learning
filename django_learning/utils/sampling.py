from __future__ import print_function

import pandas as pd

from tqdm import tqdm

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
    document_filters=None,
):

    frame_ids = set(list(samples.values_list("frame_id", flat=True)))
    if len(frame_ids) == 1:
        sampling_frame = get_model(
            "SamplingFrame", app_name="django_learning"
        ).objects.get(pk__in=frame_ids)
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

    frame = sampling_frame.get_sampling_flags(
        refresh=refresh_flags, sampling_search_subset=keyword_weight_columns
    )
    if len(keyword_weight_columns) > 0:
        keyword_weight_columns = set(
            ["search_{}".format(s) for s in keyword_weight_columns]
        )
        keyword_weight_columns.add("search_none")
    if document_filters:
        # print("Applying frame document filter: {}".format(len(frame)))
        frame = frame.rename(columns={"pk": "document_id"})
        for filter_name, filter_args, filter_kwargs in document_filters:
            frame = dataset_document_filters[filter_name](
                None, frame, *filter_args, **filter_kwargs
            )
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
        additional_weight_columns = additional_weight_columns.union(
            set(dummies.columns)
        )
        frame = frame.join(dummies)

    if "search_none" in keyword_weight_columns:
        actual_keywords = [
            k for k in list(keyword_weight_columns) if k != "search_none"
        ]
        frame["search_none"] = ~frame[actual_keywords].max(axis=1)
    full_sample = frame[
        frame["pk"].isin(
            list(
                get_model("SampleUnit", app_name="django_learning")
                .objects.filter(sample__in=samples)
                .values_list("document_id", flat=True)
            )
        )
    ].reset_index()

    if len(keyword_weight_columns) > 0:
        keyword_weight = compute_sample_weights_from_frame(
            frame, full_sample, list(keyword_weight_columns)
        )
        full_sample["keyword_weight"] = keyword_weight
    else:
        full_sample["keyword_weight"] = None

    if len(strat_weight_columns) > 0:
        strat_weight = compute_sample_weights_from_frame(
            frame, full_sample, list(strat_weight_columns)
        )
        full_sample["strat_weight"] = strat_weight
    else:
        full_sample["strat_weight"] = None

    if len(additional_weight_columns) > 0:
        additional_weight = compute_sample_weights_from_frame(
            frame, full_sample, list(additional_weight_columns)
        )
        full_sample["additional_weight"] = additional_weight
    else:
        full_sample["additional_weight"] = None

    full_sample["approx_weight"] = 1.0
    if len(keyword_weight_columns) > 0:
        full_sample["approx_weight"] *= keyword_weight
    if len(strat_weight_columns) > 0:
        full_sample["approx_weight"] *= strat_weight
    if len(additional_weight_columns) > 0:
        full_sample["approx_weight"] *= additional_weight

    all_weight_columns = keyword_weight_columns.union(strat_weight_columns).union(
        additional_weight_columns
    )
    full_weight = compute_sample_weights_from_frame(
        frame, full_sample, list(all_weight_columns)
    )
    full_sample["weight"] = full_weight
    full_sample["weight"] = full_sample["weight"].fillna(1.0)

    return full_sample


def update_frame_and_expand_samples(frame_name):

    frame_obj = get_model("SamplingFrame", app_name="django_learning").objects.get(name=frame_name)
    old_frame = frame_obj.get_sampling_flags()
    frame_obj.extract_documents(refresh=True)
    new_frame = frame_obj.get_sampling_flags(refresh=True)

    for s in frame_obj.samples.all():

        print(s)
        s.sync_with_frame()
        params = s.get_params()

        weight_vars = []
        if (
                "sampling_searches" in params.keys()
                and len(params["sampling_searches"].keys()) > 0
        ):
            weight_vars.append("search_none")
            weight_vars.extend(
                [
                    "search_{}".format(name)
                    for name in params["sampling_searches"].keys()
                ]
            )

        new_frame = frame_obj.get_sampling_flags(
            sampling_search_subset=params.get("sampling_searches", None),
            refresh=False,
        )

        if params.get("stratify_by", None):
            dummies = pd.get_dummies(new_frame[params.get("stratify_by")], prefix=params.get("stratify_by", None))
            weight_vars.extend(dummies.columns)
            new_frame = new_frame.join(dummies)

        old_frame['count'] = 1
        new_frame['count'] = 1
        existing_doc_ids = s.documents.values_list("pk", flat=True)
        sample = new_frame[new_frame['pk'].isin(existing_doc_ids)]

        EXPANSION_PCT = ((len(new_frame) - len(old_frame)) / len(old_frame))
        NEW_SAMPLE_SIZE = round(EXPANSION_PCT * len(sample))
        print(
            "Sampling frame expanded by {}, adding {} new cases to sample of {}".format(EXPANSION_PCT, NEW_SAMPLE_SIZE,
                                                                                        len(sample)))

        if len(weight_vars) > 0:

            frame_counts = new_frame.groupby(weight_vars)['count'].sum()
            frame_pct = frame_counts / new_frame['count'].sum()
            sample_counts = sample.groupby(weight_vars)['count'].sum()
            sample_pct = sample_counts / sample['count'].sum()
            pcts = pd.concat([frame_pct, frame_counts, sample_pct, sample_counts], axis=1).fillna(0.0)
            pcts.columns = ["frame_pct", "frame_count", "sample_pct", "sample_count"]
            pcts['num_to_add'] = ((pcts['frame_pct'] * (NEW_SAMPLE_SIZE + len(sample))).round()) - pcts['sample_count']
            pcts['num_to_add'] = pcts['num_to_add'].map(lambda x: max([0.0, x]))
            for strata, rows in new_frame.groupby(weight_vars):
                sample = pd.concat([
                    sample,
                    rows[~rows['pk'].isin(existing_doc_ids)].sample(int(pcts.loc[strata]['num_to_add']))
                ])

        else:
            sample = pd.concat([
                sample,
                new_frame[~new_frame['pk'].isin(existing_doc_ids)].sample(int(NEW_SAMPLE_SIZE))
            ])

        sample['weight'] = list(
            compute_sample_weights_from_frame(new_frame, sample, weight_vars)
        )

        for index, row in tqdm(sample.iterrows(), desc="Updating sample documents"):
            get_model("SampleUnit", app_name="django_learning").objects.create_or_update(
                {"document_id": row["pk"], "sample": s},
                {"weight": row["weight"]},
                return_object=False,
                save_nulls=False,
            )

#         frame_counts = new_frame.groupby(weight_vars)['count'].sum()
#         frame_pct = frame_counts / new_frame['count'].sum()
#         sample_counts = sample.groupby(weight_vars)['count'].sum()
#         sample_pct = sample_counts / sample['count'].sum()
#         pcts = pd.concat([frame_pct, frame_counts, sample_pct, sample_counts], axis=1).fillna(0.0)
#         pcts.columns = ["frame_pct", "frame_count", "sample_pct", "sample_count"]