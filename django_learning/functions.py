import pandas

from django.db.models import Q

from pewanalytics.stats.sampling import compute_sample_weights_from_frame
from django_pewtils import get_model
from pewtils import is_not_null

from django_learning.utils import filter_queryset_by_params
from django_learning.utils.dataset_document_filters import dataset_document_filters


def filter_hits(
    project=None,
    sample=None,
    turk_only=False,
    experts_only=False,
    finished_only=False,
    unfinished_only=False,
    assignments=None,
    exclude_coders=None,
    filter_coders=None,
    documents=None
):

    hits = get_model("HIT", app_name="django_learning").objects.all()
    if project: hits = hits.filter(sample__project=project)
    if sample: hits = hits.filter(sample=sample)
    if turk_only: hits = hits.filter(turk=True)
    elif experts_only: hits = hits.filter(turk=False)
    if finished_only:
        hits = get_model("HIT", app_name="django_learning").objects.filter(pk__in=[h.pk for h in hits if h.assignments.filter(time_finished__isnull=False).count() >= h.num_coders])
    elif unfinished_only:
        hits = get_model("HIT", app_name="django_learning").objects.filter(pk__in=[h.pk for h in hits if h.assignments.filter(time_finished__isnull=False).count() < h.num_coders])
    if exclude_coders != None: hits = hits.exclude(assignments__coder__in=exclude_coders)
    if filter_coders != None: hits = hits.filter(assignments__coder__in=filter_coders)
    if assignments != None: hits = hits.filter(assignments__in=assignments)
    if documents != None: hits = hits.filter(sample_unit__document__in=documents)

    return hits.distinct()


def filter_assignments(
    project=None,
    sample=None,
    turk_only=False,
    experts_only=False,
    coder_min_hit_count=None,
    coder=None,
    completed_only=False,
    incomplete_only=False,
    hits=None,
    exclude_coders=None,
    filter_coders=None,
    documents=None
):

    assignments = get_model("Assignment", app_name="django_learning").objects.all()
    if project: assignments = assignments.filter(hit__sample__project=project)
    if sample: assignments = assignments.filter(hit__sample=sample)
    if turk_only: assignments = assignments.filter(hit__turk=True)
    elif experts_only: assignments = assignments.filter(hit__turk=False)
    if coder_min_hit_count: assignments = assignments.filter(coder__in=filter_coders(project, sample=sample, min_hit_count=coder_min_hit_count))
    if coder: assignments = assignments.filter(coder=coder)
    if completed_only: assignments = assignments.filter(time_finished__isnull=False)
    elif incomplete_only: assignments = assignments.filter(time_finished__isnull=True)
    if exclude_coders != None: assignments = assignments.exclude(coder__in=exclude_coders)
    if filter_coders != None: assignments = assignments.filter(coder__in=filter_coders)
    if hits != None: assignments = assignments.filter(hit__in=hits)
    if documents != None: assignments = assignments.filter(hits__sample_unit__document__in=documents)

    return assignments.distinct()


# def filter_responses(
#     project=None,
#     sample=None,
#     turk_only=False,
#     experts_only=False,
#     coder_min_hit_count=None,
#     question=None,
#     coder=None,
#     assignments=None,
#     hits=None,
#     exclude_coders=None,
#     documents=None
# ):
#
#     responses = Response.objects.exclude(question__display="header")
#     if project: responses = responses.filter(assignment__hit__sample__project=project)
#     if sample: responses = responses.filter(assignment__hit__sample=sample)
#     if turk_only: responses = responses.filter(assignment__hit__turk=True)
#     elif experts_only: responses = responses.filter(assignment__hit__turk=False)
#     if coder_min_hit_count: responses = responses.filter(assignment__coder__in=filter_coders(project, sample=sample, min_hit_count=coder_min_hit_count))
#     if coder: responses = responses.filter(assignment__coder=coder)
#     if question: responses = responses.filter(question=question)
#     if exclude_coders != None: responses = responses.exclude(assignment__coder__in=exclude_coders)
#     if assignments != None: responses = responses.filter(assignment__in=assignments)
#     if hits != None: responses = responses.filter(assignment__hit__in=hits)
#     if documents != None: responses = responses.filter(assignment__hit__sample_unit__document__in=documents)
#
#     return responses.distinct()


def filter_coders(project=None, sample=None, min_hit_count=None):

    good_coder_ids = []
    coders = get_model("Coder", app_name="django_learning").objects.all()
    if project: coders = project.coders.all()
    if sample: coders = coders.filter(assignments__hit__sample=sample)
    for c in coders:
        a = c.assignments\
            .filter(hit__sample__project=project)\
            .filter(time_finished__isnull=False)
        if sample: a = a.filter(hit__sample=sample)
        if not min_hit_count or a.count() >= min_hit_count:
            good_coder_ids.append(c.pk)

    return get_model("Coder", app_name="django_learning").objects.filter(pk__in=good_coder_ids).distinct()


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
            for search_name in sampling_searches.keys():
                keyword_weight_columns.add(search_name)
        for additional_var in params.get("additional_weights", {}).keys():
            additional_vars.add(additional_var)

    frame = sampling_frame.get_sampling_flags(refresh=refresh_flags, sampling_search_subset=keyword_weight_columns)
    if len(keyword_weight_columns) > 0:
        keyword_weight_columns = set(["search_{}".format(s) for s in keyword_weight_columns])
        keyword_weight_columns.add("search_none")
    if document_filters:
        print "Applying frame document filter: {}".format(len(frame))
        frame = frame.rename(columns={"pk": "document_id"})
        for filter_name, filter_args, filter_kwargs in document_filters:
            frame = dataset_document_filters[filter_name](None, frame, *filter_args, **filter_kwargs)
        frame = frame.rename(columns={"document_id": "pk"})
        print "Frame is now {}".format(len(frame))
    # if filter_params:
    #     frame = frame[frame["pk"].isin(
    #         filter_queryset_by_params(sampling_frame.documents.all(), filter_params).values_list("pk", flat=True)
    #     )]

    if ignore_stratification_weights:
        strat_vars = set()

    strat_weight_columns = set()
    for stratify_by in strat_vars:
        dummies = pandas.get_dummies(frame[stratify_by], prefix=stratify_by)
        strat_weight_columns = strat_weight_columns.union(set(dummies.columns))
        frame = frame.join(dummies)

    additional_weight_columns = set()
    for additional in additional_vars:
        dummies = pandas.get_dummies(frame[additional], prefix=additional)
        additional_weight_columns = additional_weight_columns.union(set(dummies.columns))
        frame = frame.join(dummies)

    if "search_none" in keyword_weight_columns:
        actual_keywords = [k for k in list(keyword_weight_columns) if k != "search_none"]
        frame['search_none'] = ~frame[actual_keywords].max(axis=1)
    full_sample = frame[frame['pk'].isin(list(
        get_model("SampleUnit", app_name="django_learning").objects.filter(sample__in=samples).values_list(
            "document_id", flat=True)))]

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


# # def compute_median_assignment_time(assignments):
# #
# #     assignments = assignments.filter(time_finished__isnull=False)
# #     if assignments.count() > 0:
# #         df = pandas.DataFrame.from_records(assignments.values("time_finished", "time_started"))
# #         df['time_diff'] = df.apply(lambda x: (x["time_finished"] - x["time_started"]).seconds, axis=1)
# #         return df.median()['time_diff']
# #     else:
# #         return 0.0
# #
# #
# # def get_response_task_frame(responses, binary_code=None):
# #
# #     rows = []
# #     for r in responses:
# #         row = {
# #             "coder_name": r.assignment.coder.name,
# #             "obv_id": r.assignment.hit.observation_id,
# #         }
# #         if binary_code: row["code_repr"] = 1 if binary_code in r.codes.all() else 0
# #         else: row["code_repr"] = str(sorted(r.codes.values_list("pk", flat=True)))
# #         rows.append(row)
# #
# #     return pandas.DataFrame.from_records(rows)
# #
# #
# # def compute_krippendorf(df):
# #
# #     task = AnnotationTask(data=df[["coder_name", "obv_id", "code_repr"]].as_matrix())
# #     try: return task.alpha()
# #     except ZeroDivisionError: return None
# #
# #
# # def compute_kappa(df, use_pairwise=True):
# #
# #     if use_pairwise:
# #         kappas = []
# #         for c1, c2 in itertools.combinations(df['coder_name'].unique(), 2):
# #             obv_ids = set.intersection(*[set(df[df['coder_name'] == c]['obv_id'].values) for c in [c1, c2]])
# #             subset = df[df['obv_id'].isin(obv_ids)]
# #             subtask = AnnotationTask(data=subset[["coder_name", "obv_id", "code_repr"]].as_matrix())
# #             try: kappas.append(subtask.kappa_pairwise(c1, c2))
# #             except ZeroDivisionError: pass
# #         kappas = [k for k in kappas if is_not_null(k)]
# #         return numpy.average(kappas)
# #     else:
# #         try:
# #             task = AnnotationTask(data=df[["coder_name", "obv_id", "code_repr"]].as_matrix())
# #             return task.kappa()
# #         except ZeroDivisionError: return None
# #
# #
# # def compute_percent_agreement(df, use_pairwise=True):
# #
# #     if use_pairwise:
# #         percents = []
# #         for c1, c2 in itertools.combinations(df['coder_name'].unique(), 2):
# #             obv_ids = set.intersection(*[set(df[df['coder_name'] == c]['obv_id'].values) for c in [c1, c2]])
# #             subset = df[df['obv_id'].isin(obv_ids)]
# #             subtask = AnnotationTask(data=subset[["coder_name", "obv_id", "code_repr"]].as_matrix())
# #             try: percents.append(subtask.Ao(c1, c2))
# #             except ZeroDivisionError: pass
# #         percents = [k for k in percents if is_not_null(k)]
# #         return numpy.average(percents)
# #     else:
# #         try:
# #             task = AnnotationTask(data=df[["coder_name", "obv_id", "code_repr"]].as_matrix())
# #             return task.avg_Ao()
# #         except ZeroDivisionError: return None
# #
# #
# # def remove_hits_with_base_code_consensus(hits, question):
# #
# #     question_mode = question.responses\
# #         .filter(assignment__hit__in=hits)\
# #         .values("codes")\
# #         .annotate(c=Count("pk"))\
# #         .order_by("-c")
# #     if question_mode.count() > 0:
# #         question_mode = question_mode[0]['codes']
# #         good_hit_ids = []
# #         for h in hits.all():
# #             if filter_responses(hits=[h], question=question).exclude(codes__pk=question_mode).count() > 0:
# #                 good_hit_ids.append(h.pk)
# #         return hits.filter(pk__in=good_hit_ids)
# #     else:
# #         return hits.filter(pk__in=[])
# #
# # def remove_assignments_from_hits_with_base_code_consensus(assignments, question):
# #
# #     return assignments.filter(pk__in=
# #          remove_hits_with_base_code_consensus(
# #              HIT.objects.filter(pk__in=assignments.values_list("hit_id", flat=True)),
# #              question
# #          ).values_list("assignments__pk", flat=True)
# #      )
# #
# # def remove_responses_from_hits_with_base_code_consensus(responses, question):
# #
# #     return responses.filter(pk__in=
# #         remove_hits_with_base_code_consensus(
# #              HIT.objects.filter(pk__in=responses.values_list("assignment__hit_id", flat=True)),
# #              question
# #         ).values_list("assignments__responses__pk", flat=True)
# #     )