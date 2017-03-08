# import pandas, numpy, itertools, os, datetime
#
# from scipy import stats as scipy_stats
# from tqdm import tqdm
# from collections import defaultdict
#
# from django.db.models import Avg
#
# from limecoder.utils import is_not_null, is_null
# from limecoder.functions import *
# from limecoder.settings import PROJECT_FILE_ROOT
#
#
# def cache_export(func):
#     def wrapper(project, **options):
#         sample = options.get("sample", None)
#         if sample:
#             sample_folder = os.path.join(PROJECT_FILE_ROOT, project.name, "exports", sample.name)
#             if not os.path.exists(sample_folder):
#                 os.mkdir(sample_folder)
#             filepath = os.path.join(sample_folder, "{}.csv".format(func.__name__))
#         else:
#             filepath = os.path.join(PROJECT_FILE_ROOT, project.name, "exports", "{}.csv".format(func.__name__))
#         if os.path.exists(filepath) and not options.get('refresh', False):
#             return pandas.read_csv(filepath, encoding="utf8")
#         elif not options.get("cache_only", False):
#             df = func(project, **options)
#             df.to_csv(filepath, encoding="utf8")
#         else:
#             df = None
#         return df
#
#     return wrapper
#
#
# DATAFRAME_NAMES = [
#     "observations",
#     "detailed_responses",
#     "hit_question_summaries",
#     "categorical_question_summaries",
#     "categorical_code_summaries",
#     "continuous_question_summaries",
#     "coders",
#     "coder_expert_consensus_comparison"
# ]
#
# MIN_CODER_HITS = 0
#
#
# def get_dataframe_time(dataframe_name, project, sample=None):
#     if sample:
#         filepath = os.path.join(PROJECT_FILE_ROOT, project.name, "exports", sample.name, "{}.csv".format(dataframe_name))
#     else:
#         filepath = os.path.join(PROJECT_FILE_ROOT, project.name, "exports", "{}.csv".format(dataframe_name))
#     try:
#         t = os.path.getmtime(filepath)
#         return datetime.datetime.fromtimestamp(t)
#     except:
#         return None
#
#
# @cache_export
# def observations(project, sample=None, refresh=False):
#     obvs = Observation.objects.filter(samples__project=project)
#     if sample: obvs = obvs.filter(samples=sample)
#
#     rows = []
#     for o in obvs.all():
#         rows.append({
#             "id": o.pk,
#             "external_id": o.external_id,
#             "content": o.content
#         })
#
#     return pandas.DataFrame.from_records(rows)
#
#
# @cache_export
# def detailed_responses(project, sample=None, refresh=False):
#     responses = filter_responses(project=project, sample=sample, coder_min_hit_count=MIN_CODER_HITS)
#     qual_responses = filter_responses(project=project, sample=sample, qualification=True,
#                                       coder_min_hit_count=MIN_CODER_HITS)
#
#     rows = []
#     for r in tqdm(itertools.chain(responses, qual_responses), desc="Computing assignment responses", leave=True):
#         # codes = sorted(r.codes.values_list("value", flat=True))
#         # if len(codes) == 1: codes = codes[0]
#         for code in r.codes.all():
#             rows.append({
#                 "response": r,
#                 "response_id": r.pk,
#                 "sample_name": sample.name if sample else None,
#                 "hit_id": r.assignment.hit.pk,
#                 "hit_turk": r.assignment.hit.turk,
#                 "is_qualification_hit": True if r.assignment.hit.qualification_hit_type else False,
#                 "assignment_id": r.assignment.pk,
#                 "observation_id": r.assignment.hit.observation.pk if r.assignment.hit.observation else None,
#                 "external_id": r.assignment.hit.observation.external_id if r.assignment.hit.observation else None,
#                 "question_name": r.question.name,
#                 "coder_id": r.assignment.coder.pk,
#                 "coder_name": r.assignment.coder.name,
#                 "code": code.value,
#                 "assignment_response_time": (
#                 r.assignment.time_finished - r.assignment.time_started).seconds if r.assignment.time_finished else None
#             })
#
#     return pandas.DataFrame.from_records(rows)
#
#
# @cache_export
# def hit_question_summaries(project, sample=None, refresh=False):
#     hits = filter_hits(project=project, sample=sample)
#     qual_hits = filter_hits(project=project, sample=sample, qualification=True)
#
#     rows = []
#     for h in tqdm(itertools.chain(hits, qual_hits), desc="Computing HIT responses", leave=True):
#         assignments = filter_assignments(project=project, sample=sample, hits=[h], incomplete=False,
#                                          coder_min_hit_count=MIN_CODER_HITS)
#         # assignments = h.assignments.filter(time_finished__isnull=False)
#         completed = assignments.count()
#         median_response_time = compute_median_assignment_time(assignments)
#         for q in project.questions.filter(qualification=True if h.qualification_hit_type else False).exclude(
#                 display="header"):
#             codes = []
#             # for r in Response.objects.filter(assignment__hit=h, question=q):
#             for r in filter_responses(project=project, sample=sample, hits=[h], question=q,
#                                       coder_min_hit_count=MIN_CODER_HITS):
#                 c = sorted(r.codes.values_list("label", flat=True))
#                 if len(c) == 1:
#                     c = c[0]
#                 else:
#                     c = str(c)
#                 codes.append(c)
#             mode = scipy_stats.mode(numpy.array(codes)).mode[0]
#             rows.append({
#                 "hit": h,
#                 "hit_id": h.pk,
#                 "observation_id": h.observation.pk if h.observation else None,
#                 "external_id": h.observation.external_id if h.observation else None,
#                 "question_name": q.name,
#                 "sample_name": sample.name if sample else None,
#                 "hit_turk": h.turk,
#                 "is_qualification_hit": True if h.qualification_hit_type else False,
#                 "completed_assignments": completed,
#                 "total_assignments_requested": h.num_coders,
#                 "is_finished": completed == h.num_coders,
#                 "mode": mode,
#                 "hit_median_assignment_time": median_response_time,
#                 "percent_mode": (float(len([c for c in codes if c == mode])) / float(len(codes))) if len(
#                     codes) > 0 else None
#             })
#
#     return pandas.DataFrame.from_records(rows)
#
#
# @cache_export
# def categorical_code_summaries(project, sample=None, refresh=False):
#     rows = []
#     for q in tqdm(project.questions.exclude(display="number").exclude(display="header"),
#                   desc="Computing categorical question code summaries", leave=True):
#
#         expert_responses = filter_responses(project=project, sample=sample, experts_only=True,
#                                             qualification=q.qualification, question=q,
#                                             coder_min_hit_count=MIN_CODER_HITS)
#         turk_responses = filter_responses(project=project, sample=sample, turk_only=True, qualification=q.qualification,
#                                           question=q, coder_min_hit_count=MIN_CODER_HITS)
#
#         expert_code_counts = {r["assignment__hit__observation_id"]: r["c"] for r in
#                               expert_responses.values("assignment__hit__observation_id").annotate(
#                                   c=Count("pk")).distinct()}
#         turk_code_counts = {r["assignment__hit__observation_id"]: r["c"] for r in
#                             turk_responses.values("assignment__hit__observation_id").annotate(
#                                 c=Count("pk")).distinct()}
#
#         for c in q.codes.all():
#
#             expert_obv_counts = {}
#             for r in expert_responses.filter(codes=c).values("assignment__hit__observation_id").annotate(
#                     c=Count("pk")).distinct():
#                 if is_not_null(r['c']):
#                     expert_obv_counts[r["assignment__hit__observation_id"]] = float(r["c"])
#             avg_expert_code_pct = numpy.average(
#                 [expert_obv_counts.get(oid, 0.0) / float(val) for oid, val in expert_code_counts.iteritems()])
#
#             turk_obv_counts = {}
#             for r in turk_responses.filter(codes=c).values("assignment__hit__observation_id").annotate(
#                     c=Count("pk")).distinct():
#                 if is_not_null(r['c']):
#                     turk_obv_counts[r["assignment__hit__observation_id"]] = float(r["c"])
#             avg_turk_code_pct = numpy.average(
#                 [turk_obv_counts.get(oid, 0.0) / float(val) for oid, val in turk_code_counts.iteritems()])
#
#             row = {
#                 "question": q,
#                 "code": c,
#                 "question_name": q.name,
#                 "is_qualification_question": q.qualification,
#                 "code_label": c.label,
#                 "code_value": c.value,
#                 "sample_name": sample.name if sample else None,
#                 "expert_avg_percent_observations": avg_expert_code_pct,
#                 "turk_avg_percent_observations": avg_turk_code_pct,
#                 "expert_count": expert_responses.filter(codes=c).count(),
#                 "turk_count": turk_responses.filter(codes=c).count()
#             }
#             rows.append(row)
#
#     return pandas.DataFrame.from_records(rows)
#
#
# @cache_export
# def categorical_question_summaries(project, sample=None, refresh=False):
#     rows = []
#     for q in tqdm(project.questions.exclude(display="number").exclude(display="header"),
#                   desc="Computing categorical question summaries", leave=True):
#
#         expert_responses = filter_responses(project=project, sample=sample, experts_only=True,
#                                             qualification=q.qualification, question=q,
#                                             coder_min_hit_count=MIN_CODER_HITS)
#         turk_responses = filter_responses(project=project, sample=sample, turk_only=True,
#                                           qualification=q.qualification, question=q, coder_min_hit_count=MIN_CODER_HITS)
#
#         expert_code_frequencies = sorted([(c, expert_responses.filter(codes=c).count()) for c in q.codes.all()],
#                                          key=lambda x: x[1])
#         turk_code_frequencies = sorted([(c, turk_responses.filter(codes=c).count()) for c in q.codes.all()],
#                                        key=lambda x: x[1])
#
#         row = {
#             "question": q,
#             "question_name": q.name,
#             "is_qualification_question": q.qualification,
#             "sample_name": sample.name if sample else None,
#             "expert_most_frequent_code": expert_code_frequencies[-1][0].label,
#             "expert_most_frequent_code_count": expert_code_frequencies[-1][1],
#             "expert_least_frequent_code": expert_code_frequencies[0][0].label,
#             "expert_least_frequent_code_count": expert_code_frequencies[0][1],
#             "turk_most_frequent_code": turk_code_frequencies[-1][0].label,
#             "turk_most_frequent_code_count": turk_code_frequencies[-1][1],
#             "turk_least_frequent_code": turk_code_frequencies[0][0].label,
#             "turk_least_frequent_code_count": turk_code_frequencies[0][1]
#         }
#
#         if not q.qualification:
#             expert_task_frame = get_response_task_frame(
#                 expert_responses  # remove_responses_from_hits_with_base_code_consensus(expert_responses, q)
#             )
#             turk_task_frame = get_response_task_frame(
#                 turk_responses  # remove_responses_from_hits_with_base_code_consensus(turk_responses, q)
#             )
#
#             row.update({
#                 "expert_krippendorf": compute_krippendorf(expert_task_frame) if len(expert_task_frame) > 0 else None,
#                 "turk_krippendorf": compute_krippendorf(turk_task_frame) if len(turk_task_frame) > 0 else None,
#                 "expert_kappa": compute_kappa(expert_task_frame) if len(expert_task_frame) > 0 else None,
#                 "turk_kappa": compute_kappa(turk_task_frame) if len(turk_task_frame) > 0 else None,
#                 "expert_percent_agreement": compute_percent_agreement(expert_task_frame) if len(
#                     expert_task_frame) > 0 else None,
#                 "turk_percent_agreement": compute_percent_agreement(turk_task_frame) if len(
#                     turk_task_frame) > 0 else None,
#             })
#
#         rows.append(row)
#
#     return pandas.DataFrame.from_records(rows)
#
#
# @cache_export
# def continuous_question_summaries(project, sample=None, refresh=False):
#     rows = []
#     for q in tqdm(project.questions.filter(display="number").exclude(display="header"),
#                   desc="Computing continuous question summaries", leave=True):
#
#         expert_responses = filter_responses(project=project, sample=sample, experts_only=True,
#                                             qualification=q.qualification, question=q,
#                                             coder_min_hit_count=MIN_CODER_HITS)
#         turk_responses = filter_responses(project=project, sample=sample, turk_only=True, qualification=q.qualification,
#                                           question=q, coder_min_hit_count=MIN_CODER_HITS)
#
#         expert_codes = [float(c.split()[0]) for c in expert_responses.values_list("codes__value", flat=True)]
#         turk_codes = [float(c.split()[0]) for c in turk_responses.values_list("codes__value", flat=True)]
#
#         row = {
#             "question_name": q.name,
#             "sample_name": sample.name if sample else None,
#             "expert_responses": expert_responses.count(),
#             "turk_responses": turk_responses.count(),
#             "turk_min_code": min(turk_codes) if len(turk_codes) > 0 else None,
#             "expert_min_code": min(expert_codes) if len(expert_codes) > 0 else None,
#             "turk_max_code": max(turk_codes) if len(turk_codes) > 0 else None,
#             "expert_max_code": max(expert_codes) if len(expert_codes) > 0 else None,
#             "turk_mean_code": numpy.average(turk_codes) if len(turk_codes) > 0 else None,
#             "expert_mean_code": numpy.average(expert_codes) if len(expert_codes) > 0 else None,
#             "turk_median_code": numpy.median(turk_codes) if len(turk_codes) > 0 else None,
#             "expert_median_code": numpy.median(expert_codes) if len(expert_codes) > 0 else None,
#         }
#
#         if not q.qualification:
#             expert_task_frame = get_response_task_frame(
#                 expert_responses  # remove_responses_from_hits_with_base_code_consensus(expert_responses, q)
#             )
#             turk_task_frame = get_response_task_frame(
#                 turk_responses  # remove_responses_from_hits_with_base_code_consensus(turk_responses, q)
#             )
#
#             row.update({
#                 "expert_krippendorf": compute_krippendorf(expert_task_frame),
#                 "turk_krippendorf": compute_krippendorf(turk_task_frame),
#                 "expert_kappa": compute_kappa(expert_task_frame),
#                 "turk_kappa": compute_kappa(turk_task_frame),
#                 "expert_percent_agreement": compute_percent_agreement(expert_task_frame),
#                 "turk_percent_agreement": compute_percent_agreement(turk_task_frame)
#             })
#
#         rows.append(row)
#
#     return pandas.DataFrame.from_records(rows)
#
#
# @cache_export
# def coder_expert_consensus_comparison(project, sample=None, refresh=False):
#     rows = []
#
#     for q in tqdm(project.questions.filter(qualification=False), desc="Iterating over questions"):
#
#         expert_responses = filter_responses(project=project, sample=sample, experts_only=True, qualification=False,
#                                             coder_min_hit_count=MIN_CODER_HITS, question=q)
#         # expert_responses = remove_responses_from_hits_with_base_code_consensus(expert_responses, q)
#
#         consensus_obvs = {}
#         for obv in expert_responses.values("assignment__hit__observation_id").annotate(avg=Avg("codes")):
#             if float(int(obv["avg"])) == float(obv["avg"]):
#                 consensus_obvs[obv["assignment__hit__observation_id"]] = obv['avg']
#
#         observations = Observation.objects.filter(pk__in=consensus_obvs.keys())
#         for c in tqdm(project.coders.filter(turk=True), nested=True, desc="Iterating over Turk coders"):
#             turk_responses = filter_responses(project=project, sample=sample, turk_only=True, qualification=False,
#                                               question=q, coder_min_hit_count=MIN_CODER_HITS, coder=c,
#                                               observations=observations)
#             # turk_responses = remove_responses_from_hits_with_base_code_consensus(turk_responses, q)
#             if turk_responses.count() > 0:
#                 agreements = []
#                 for r in turk_responses.all():
#                     if float(numpy.average(list(r.codes.values_list("pk", flat=True)))) == float(
#                             consensus_obvs[r.assignment.hit.observation_id]):
#                         agreements.append(1.0)
#                     else:
#                         agreements.append(0.0)
#
#                 rows.append({
#                     "coder_name": c.name,
#                     "question_id": q.pk,
#                     "question_name": q.name,
#                     "pct_expert_consensus_correct": numpy.average(agreements),
#                     "num_obvs_in_common": len(agreements)
#                 })
#
#     return pandas.DataFrame.from_records(rows)
#
#
# @cache_export
# def coders(project, sample=None, refresh=False):
#     coders = filter_coders(project=project, sample=sample, min_hit_count=MIN_CODER_HITS)
#     # min_count_coders = filter_coders(project=project, sample=sample, min_hit_count=10)
#
#     rows = []
#     for c in tqdm(coders.all(), desc="Computing coder stats", leave=True):
#         assignments = filter_assignments(project=project, sample=sample, coder=c, incomplete=False)
#         median_time = compute_median_assignment_time(assignments)
#         qual_responses = filter_responses(project=project, sample=sample, coder=c, qualification=True)
#         row = {
#             "coder_id": c.pk,
#             "coder_name": c.name,
#             "turk": c.turk,
#             "assignments_completed": assignments.count(),
#             "median_assignment_time": median_time
#         }
#         for r in qual_responses:
#             codes = sorted(r.codes.values_list("value", flat=True))
#             if len(codes) == 1: codes = codes[0]
#             row["qual_{}".format(r.question.name)] = codes
#
#         # if c in min_count_coders:
#         agreements = defaultdict(list)
#         for q in project.questions.filter(qualification=False).exclude(display="header"):
#             c_responses = filter_responses(project=project, sample=sample, coder=c, question=q)
#             # c_responses = remove_responses_from_hits_with_base_code_consensus(c_responses, q)
#             for c2 in coders.exclude(pk=c.pk):
#                 # if c2 in min_count_coders:
#                 c2_responses = filter_responses(project=project, sample=sample, coder=c2, question=q)
#                 # c2_responses = remove_responses_from_hits_with_base_code_consensus(c2_responses, q)
#                 task_frame = get_response_task_frame(itertools.chain(c_responses, c2_responses))
#                 if len(task_frame) > 0:
#                     if not c2.turk:
#                         agreements["krippendorf_experts"].append(compute_krippendorf(task_frame))
#                         agreements["kappa_experts"].append(compute_kappa(task_frame))
#                         agreements["pct_agree_experts"].append(compute_percent_agreement(task_frame))
#                     else:
#                         agreements["krippendorf_turk"].append(compute_krippendorf(task_frame))
#                         agreements["kappa_turk"].append(compute_kappa(task_frame))
#                         agreements["pct_agree_turk"].append(compute_percent_agreement(task_frame))
#         for metric in [
#             "krippendorf_experts",
#             "kappa_experts",
#             "pct_agree_experts",
#             "krippendorf_turk",
#             "kappa_turk",
#             "pct_agree_turk"
#         ]:
#             if len(agreements[metric]) > 0:
#                 row[metric] = numpy.average([m for m in agreements[metric] if m])
#
#         rows.append(row)
#
#     return pandas.DataFrame.from_records(rows)
#
#
#
#
#
#     # def coder_pairs(context):
#     #
#     #     rows = []
#     #     scanned_coders = []
#     #     for c1 in tqdm(context.coders.all(), desc="Computing coder pair stats"):
#     #         scanned_coders.append(c1.pk)
#     #         context1 = CoderContext(c1, context)
#     #         for c2 in context.coders.exclude(pk__in=scanned_coders):
#     #             row = {
#     #                 "coder_id1": c1.pk,
#     #                 "coder_id2": c2.pk,
#     #                 "coder_name1": c1.name,
#     #                 "coder_name2": c2.name,
#     #                 "overlapping_obvs": context1.overlapping_observations(c2).count()
#     #             }
#     #             for q in context.project.questions.all():
#     #                 qcontext = QuestionContext(q, context)
#     #                 qcontext.responses = qcontext.responses.filter(assignment__coder_id__in=[c1.pk, c2.pk])
#     #                 for metric, val in qcontext.agreement_stats.iteritems():
#     #                     row["{}_{}".format(q.name, metric)] = val
#     #
#     #             rows.append(row)
#     #
#     #     return pandas.DataFrame.from_records(rows)