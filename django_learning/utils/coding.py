from __future__ import print_function

from django_pewtils import get_model


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