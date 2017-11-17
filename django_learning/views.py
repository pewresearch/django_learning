import random, datetime, os

from StringIO import StringIO

from django.shortcuts import render, redirect
from django.template import RequestContext
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate
from django.db.models import Count, F
from django.contrib.auth import login as django_login
from django.contrib.auth import logout as django_logout
from django.http import StreamingHttpResponse
from django.conf import settings

from django_commander.utils import run_command_task

from django_learning.exceptions import RequiredResponseException
from django_learning.models import *
from django_learning.functions import *
from django_learning.utils.projects import projects as project_configs
from django_learning.utils.sampling_frames import sampling_frames as sampling_frame_configs
from django_learning.utils.sampling_methods import sampling_methods as sampling_method_configs
from django_learning.utils.project_hit_types import project_hit_types as project_hit_type_configs

from pewtils import is_not_null
from pewtils.django import get_model


# def render_hit(request, hit_id):
#
#     hit = HIT.objects.get(pk=hit_id)
#     return render_to_response("hit_templates/hit.html", {
#         "hit": hit,
#         "questions": hit.questions.order_by("priority")
#     }, context_instance=RequestContext(request))

@login_required
def home(request):

    projects = request.user.coder.projects.all() | request.user.coder.admin_projects.all()
    projects = projects.distinct()
    existing_project_names = list(Project.objects.values_list("name", flat=True))
    uncreated_projects = [p for p in project_configs.keys() if p not in existing_project_names]

    sampling_frames = SamplingFrame.objects.all()
    existing_sampling_frame_names = list(SamplingFrame.objects.values_list("name", flat=True))
    uncreated_sampling_frames = [s for s in sampling_frame_configs.keys() if s not in existing_sampling_frame_names]

    return render(request, 'django_learning/index.html', {
        "projects": projects,
        "uncreated_projects": uncreated_projects,
        "sampling_frames": sampling_frames,
        "uncreated_sampling_frames": uncreated_sampling_frames
    })


@login_required
def view_project(request, project_name):

    try: project = Project.objects.get(name=project_name)
    except Project.DoesNotExist: project = None

    samples = []

    if project and request.user.coder in project.coders.all():
        for sample in project.samples.all():

            request.user.coder._clear_abandoned_sample_assignments(sample)

            samples.append({
                "sample": sample,
                "sample_units": sample.document_units.count()
            })

    return render(request, 'django_learning/project.html', {
        "project": project,
        "project_name": project_name,
        "samples": samples,
        # "hit_types": project_hit_type_configs, # TODO: shouldn't this be only HITTypes associated with the project?
        # "sampling_methods": sampling_method_configs,
        # "sampling_frames": SamplingFrame.objects.all()
        # "dataframes": dataframes.DATAFRAME_NAMES
    })


@login_required
def create_project(request, project_name):

    run_command_task.delay("create_project", {
        "project_name": project_name
    })

    return home(request)


@login_required
def edit_project_coders(request, project_name, mode):

    try: project = Project.objects.get(name=project_name)
    except Project.DoesNotExist: project = None

    if project:

        if request.method == "POST":

            active_coders = []
            inactive_coders = []
            new_name = request.POST.get("new_name")
            if new_name:
                try:
                    user = User.objects.get(username=new_name)
                except User.DoesNotExist:
                    user = User.objects.create_user(
                        new_name,
                        "{}@pewresearch.org".format(new_name),
                        "pass"
                    )
                coder = get_model("Coder").objects.create_or_update(
                    {"name": new_name},
                    {"is_mturk": False, "user": user}
                )
                status = request.POST.get("new")
                if status == "active": active_coders.append(coder.pk)
                else: inactive_coders.append(coder.pk)

            for coder_id, status in request.POST.iteritems():
                if not coder_id.startswith("new") and not coder_id.startswith("csrf"):
                    coder = Coder.objects.get(pk=coder_id)
                    if status == "active": active_coders.append(coder.pk)
                    else: inactive_coders.append(coder.pk)

            project.coders = Coder.objects.filter(pk__in=active_coders)
            project.inactive_coders = Coder.objects.filter(pk__in=inactive_coders)
            project.save()

        coders = []
        for coder in project.coders.filter(is_mturk=True if mode == 'mturk' else False):
            coders.append({
                "name": str(coder),
                "pk": coder.pk,
                "status": "active"
            })
        for coder in project.inactive_coders.filter(is_mturk=True if mode == 'mturk' else False):
            coders.append({
                "name": str(coder),
                "pk": coder.pk,
                "status": "inactive"
            })
        nonproject_coders = get_model("Coder").objects.filter(is_mturk=True if mode == 'mturk' else False)\
            .exclude(pk__in=project.coders.all())\
            .exclude(pk__in=project.inactive_coders.all())
        for coder in nonproject_coders:
            coders.append({
                "name": str(coder),
                "pk": coder.pk,
                "status": "---"
            })

        return render(request, "django_learning/edit_coders.html", {
            "project": project,
            "coders": coders,
            "mode": mode
        })

    else:

        return home(request)


@login_required
def view_sampling_frame(request, sampling_frame_name):

    return render(request, "django_learning/sampling_frame.html", {
        "sampling_frame": SamplingFrame.objects.get_if_exists({"name": sampling_frame_name}),
        "sampling_frame_name": sampling_frame_name,
        "sampling_frame_config": sampling_frame_configs[sampling_frame_name]
    })


@login_required
def extract_sampling_frame(request, sampling_frame_name):

    run_command_task.delay("extract_sampling_frame", {
        "sampling_frame_name": sampling_frame_name,
        "refresh": True
    })

    return view_sampling_frame(request, sampling_frame_name)


@login_required
def extract_sample(request, project_name):

    project = Project.objects.get(name=project_name)

    if request.user.coder in project.admins.all():
        run_command_task.delay("extract_sample", {
            "project_name": project_name,
            "hit_type_name": request.POST.get("hit_type_name"),
            "sample_name": request.POST.get("sample_name"),
            "sampling_frame_name": request.POST.get("sampling_frame_name"),
            "sampling_method": request.POST.get("sampling_method"),
            "size": int(request.POST.get("size")),
            "allow_overlap_with_existing_project_samples": bool(request.POST.get("allow_overlap_with_existing_project_samples"))
        })

    return view_project(request, project_name)


@login_required
def view_sample(request, project_name, sample_name):

    project = Project.objects.get(name=project_name)
    sample = project.samples.get(name=sample_name)

    total_expert_hits = filter_hits(sample=sample, experts_only=True)
    available_expert_hits = filter_hits(sample=sample, unfinished_only=True, experts_only=True,
                                        exclude_coders=[request.user.coder])
    completed_expert_hits = filter_hits(
        assignments=filter_assignments(sample=sample, experts_only=True, coder=request.user.coder, completed_only=True),
        experts_only=True)
    total_turk_hits = filter_hits(sample=sample, turk_only=True)
    completed_turk_hits = filter_hits(sample=sample, finished_only=True, turk_only=True)
    total_turk_assignments = sum(list([h.num_coders for h in total_turk_hits.all()]))
    completed_turk_assignments = filter_assignments(sample=sample, completed_only=True, turk_only=True)

    expert_coder_completion = []
    mturk_coder_completion = []
    if request.user.coder in project.admins.all():

        for coder in project.coders.exclude(pk=request.user.coder.pk).filter(is_mturk=False):
            assignments = sample.assignments.filter(coder=coder).count()
            if assignments > 0:
                coder_available_expert_hits = filter_hits(sample=sample, unfinished_only=True, experts_only=True, exclude_coders=[coder])
                coder_completed_expert_hits = filter_hits(
                    assignments=filter_assignments(sample=sample, experts_only=True, coder=coder, completed_only=True),
                    experts_only=True
                )
                expert_coder_completion.append({
                    "coder": coder,
                    "completed_expert_hits": coder_completed_expert_hits,
                    "available_expert_hits": coder_available_expert_hits
                })

        for coder in project.coders.exclude(pk=request.user.coder.pk).filter(is_mturk=True):
            assignments = sample.assignments.filter(coder=coder).count()
            if assignments > 0:
                coder_completed_turk_assignments = filter_assignments(sample=sample, turk_only=True, coder=coder, completed_only=True)
                coder_total_turk_assignments = filter_assignments(sample=sample, turk_only=True, coder=coder)
                mturk_coder_completion.append({
                    "coder": coder,
                    "completed_turk_assignments": coder_completed_turk_assignments,
                    "total_turk_assignments": coder_total_turk_assignments
                })
        # mturk_coder_completion = sorted(mturk_coder_completion, key=lambda x: x["completed_turk_assignments"], reverse=True)

    return render(request, "django_learning/sample.html", {
        "sample": sample,
        "total_expert_hits": total_expert_hits,
        "available_expert_hits": available_expert_hits,
        "completed_expert_hits": completed_expert_hits,
        "total_turk_hits": total_turk_hits,
        "completed_turk_hits": completed_turk_hits,
        "total_turk_assignments": total_turk_assignments,
        "completed_turk_assignments": completed_turk_assignments,
        "expert_coder_completion": expert_coder_completion,
        "mturk_coder_completion": mturk_coder_completion
    })


@login_required
def create_sample_hits_experts(request, project_name):

    project = Project.objects.get(name=project_name)

    if request.user.coder in project.admins.all():
        run_command_task.delay("create_sample_hits_experts", {
            "project_name": project_name,
            "sample_name": request.POST.get("sample_name"),
            "num_coders": int(request.POST.get("num_coders"))
        })

    return view_sample(request, project_name, request.POST.get("sample_name"))


@login_required
def create_sample_hits_mturk(request, project_name):

    project = Project.objects.get(name=project_name)

    if request.user.coder in project.admins.all():
        run_command_task.delay("create_sample_hits_mturk", {
            "project_name": project_name,
            "sample_name": request.POST.get("sample_name"),
            "num_coders": int(request.POST.get("num_coders")),
            "prod": bool(request.POST.get("prod"))
        })

    return view_sample(request, project_name, request.POST.get("sample_name"))


@login_required
def complete_qualification(request, project_name, sample_name, qualification_test_name):

    if request.method == "POST":

        qual_test = QualificationTest.objects.get(pk=request.POST.get("test_id"))
        assignment = QualificationAssignment.objects.get(test=qual_test, coder=request.user.coder)
        if not assignment.time_finished:

            assignment.time_finished = datetime.datetime.now()
            assignment.save()

            for field in request.POST.keys():
                try:
                    question = qual_test.questions.get(name=field)
                    question.update_assignment_response(assignment, request.POST.get(field))
                except Question.DoesNotExist:
                    pass

        else:

            return view_sample(request, project_name, sample_name)

    qual_test = QualificationTest.objects.get(name=qualification_test_name)
    project = Project.objects.get(name=project_name)
    if request.user.coder in project.coders.all():

        sample = Sample.objects.get(project=project, name=sample_name)
        request.user.coder._clear_abandoned_sample_assignments(sample)

        if sample.hit_type.is_qualified(request.user.coder):
            return code_assignment(request, project.name, sample.name, skip_post=True)
        else:
            QualificationAssignment.objects.create_or_update({"test": qual_test, "coder": request.user.coder}, return_object=False)
            return _render_qualification_test(request, project, sample, qual_test)

    else:

        return view_project(request, project.name)


@login_required
def code_assignment(request, project_name, sample_name, assignment_id=None, skip_post=False):

    if request.method == "POST" and not skip_post:

        try:
            saved = _save_response(request)
            if not saved:
                return view_project(request, project_name)
        except RequiredResponseException:
            return render(request, "django_learning/alert.html", {"message": "You didn't fill out all of the required responses"})

    project = Project.objects.get(name=project_name)

    if request.user.coder in project.coders.all() or request.user.coder in project.admins.all():

        sample = Sample.objects.get(project=project, name=sample_name)
        request.user.coder._clear_abandoned_sample_assignments(sample)

        qual_tests = project.qualification_tests.all() | sample.hit_type.qualification_tests.all()
        for qual_test in qual_tests:
            try: qual = QualificationAssignment.objects.filter(time_finished__isnull=False).get(test=qual_test, coder=request.user.coder)
            except QualificationAssignment.DoesNotExist: qual = None
            if not qual:
                return complete_qualification(request, project.name, sample.name, qual_test.name)

        if sample.hit_type.is_qualified(request.user.coder):

            if assignment_id:

                assignment = Assignment.objects.get(pk=assignment_id)
                return _render_assignment(request, project, sample, assignment)

            else:

                queue_ordering = request.GET.getlist("order_queue_by", ["?"])
                # queue_ordering.reverse()
                hits_available = filter_hits(sample=sample, unfinished_only=True, experts_only=True, exclude_coders=[request.user.coder])\
                    .distinct() # .annotate(c=Count("assignments"))
                hits_available = hits_available.order_by(*queue_ordering)

                if hits_available.count() > 0:

                    hit = hits_available[0]
                    assignment = Assignment.objects.create_or_update({"hit": hit, "coder": request.user.coder})
                    return _render_assignment(request, project, sample, assignment, remaining_count=len(hits_available))

                else:
                    return render(request, "django_learning/alert.html", {"message": "No available assignments for this sample!"})
                    # return view_project(request, project_name)

        else:
            # return view_project(request, project_name)
            return render(request, "django_learning/alert.html", {"message": "You're not qualified to work on these assignments."})
    else:
        # return view_project(request, project_name)
        return render(request, "django_learning/alert.html", {"message": "You're not currently registered as an active coder on this project."})


def _render_qualification_test(request, project, sample, qual_test):

    return render(request, "django_learning/qual_test.html", {
        "project": project,
        "sample": sample,
        "qual_test": qual_test
    })


def _render_assignment(request, project, sample, assignment, remaining_count=None, form_post_path=None, additional_context=None):

    if assignment.hit.template_name:
        template = "{}.html".format(assignment.hit.template_name)
        # for folder in settings.DJANGO_LEARNING_HIT_TEMPLATE_DIRS:
        #     path = os.path.join(folder, "{}.html".format(hit.template_name))
        #     if os.path.exists(path):
        #         template = path
        # # template = "custom_hits/{}.html".format(hit.template_name)
    else:
        template = "django_learning/hit.html"

    questions = []
    for q in project.questions.all():
        if assignment.codes.count() > 0:
            if q.display in ["radio", "dropdown", "checkbox"]:
                q.existing_label_ids = list(assignment.codes.filter(label__question=q).values_list("label_id", flat=True))
            elif q.display == "number":
                existing = assignment.codes.get_if_exists({"label__question": q})
                if existing: q.existing_value = existing.label.value
        questions.append(q)

    context = {
        "remaining_count": remaining_count,
        "project": project,
        "sample": sample,
        "assignment": assignment,
        "hit": assignment.hit,
        "questions": questions,
        "form_post_path": form_post_path,
        "existing_labels": assignment.codes.values_list("label_id", flat=True)
    }
    if additional_context:
        context.update(additional_context)
    return render(request, template, context)


def _save_response(request, overwrite=False):

    incomplete = False
    hit = HIT.objects.get(pk=request.POST.get("hit_id"))
    if hit.sample and request.user.coder in hit.sample.project.coders.all() \
            or request.user.coder in hit.sample.project.admins.all():

        assignment = Assignment.objects.get_if_exists({"pk": request.POST.get("assignment_id")})
        if not assignment:
            assignment = Assignment.objects.get(hit=hit, coder=request.user.coder)
        if not assignment.time_finished or overwrite:

            for field in request.POST.keys():
                try:
                    question = hit.sample.project.questions.get(name=field)
                    question.update_assignment_response(assignment, request.POST.get(field))
                except Question.DoesNotExist:
                    if field == "notes":
                        assignment.notes = request.POST.get(field)
                        assignment.save()
                    elif field == "uncodeable":
                        if int(request.POST.get(field)) == 1:
                            assignment.uncodeable = True
                            assignment.save()
                except RequiredResponseException:
                    incomplete = True
            for q in hit.sample.project.questions.exclude(name__in=request.POST.keys()):
                default_labels = q.labels.filter(select_as_default=True)
                if default_labels.count() > 0:
                    if q.multiple:
                        q.update_assignment_resposne(assignment, list(default_labels.values_list("pk", flat=True)))
                    elif default_labels.count() == 1:
                        q.update_assignment_response(assignment, default_labels[0].pk)
            if "uncodeable" not in request.POST.keys():
                assignment.uncodeable = False
                assignment.save()
            if not overwrite and not incomplete:
                assignment.time_finished = datetime.datetime.now()
                assignment.save()

        if incomplete:
            raise RequiredResponseException()

        return True

    else:

        return False

@login_required
def adjudicate_question(request, project_name, sample_name, question_name):

    project = Project.objects.get(name=project_name)
    sample = project.samples.get(name=sample_name)

    if request.user.coder in sample.project.admins.all():

        if request.method == "POST":

            hit_id = request.POST.get("hit_id")
            ignore_coder_id = request.POST.get("ignore_coder_id")

            code = HIT.objects.get(pk=hit_id).assignments.get(coder_id=ignore_coder_id).codes.get(label__question__name=question_name)
            code.consensus_ignore = True
            code.save()

        completed_expert_hits = filter_hits(
            assignments=filter_assignments(sample=sample, experts_only=True, completed_only=True),
            experts_only=True
        )

        codes = pandas.DataFrame.from_records(
            Code.objects\
                .filter(consensus_ignore=False)\
                .filter(assignment__hit__in=completed_expert_hits)\
                .filter(label__question__name=question_name)
                .values("label_id", "assignment__hit__id", "coder_id")
        )
        grouped = codes.groupby(["assignment__hit__id"]).agg({"label_id": lambda x: x.nunique()})
        hits = grouped[grouped['label_id']>1]
        hits = HIT.objects.filter(pk__in=hits.index)

        random_hit = hits.order_by("?")[0]
        codes = codes[codes['assignment__hit__id'] == random_hit.pk].groupby("label_id")['coder_id'].first()

        return render(request, "django_learning/adjudicate_question.html", {
            "question_name": question_name,
            "sample": sample,
            "hit": random_hit,
            "coder_1": Coder.objects.get(pk=codes.values[0]),
            "coder_2": Coder.objects.get(pk=codes.values[1]),
            "label_1": Label.objects.get(pk=codes.index[0]),
            "label_2": Label.objects.get(pk=codes.index[1])
        })


@login_required
def view_topic_models(request):

    topic_models = TopicModel.objects.all()
    return render(request, "django_learning/topic_models.html", {
        "topic_models": topic_models
    })


@login_required
def edit_topic_model(request, model_id):

    topic_model = TopicModel.objects.get(pk=model_id)

    if request.method == "POST":
        for topic in topic_model.topics.order_by("num"):
            name = request.POST.get("topic_{}_name".format(topic.num), None)
            topic.name = name
            label = request.POST.get("topic_{}_label".format(topic.num), None)
            topic.label = label
            anchors = request.POST.get("topic_{}_anchors".format(topic.num), None)
            if is_not_null(anchors):
                anchors = [anchor.strip() for anchor in anchors.split(",")]
                topic.anchors = anchors
            else:
                topic.anchors = []
            topic.save()

    return render(request, "django_learning/topic_model.html", {
        "topic_model": topic_model,
        "topics": topic_model.topics.order_by("num")
    })

# @login_required
# def get_dataframe(request, project_name):
#
#     project = Project.objects.get(name=project_name)
#
#     if request.user.coder in project.admins.all():
#
#         sample_name = request.POST.get("sample_name", "all")
#         sample = None
#         if sample_name != "all":
#             sample = project.samples.get(name=sample_name)
#         dataframe_name = request.POST.get("dataframe")
#         access_mode = request.POST.get("access_mode", "view")
#
#         df = getattr(dataframes, dataframe_name)(project, sample=sample, cache_only=True)
#
#         if access_mode == "view":
#             return render_to_response("admin/dataframe.html", {
#                 "project": project,
#                 "sample_name": sample_name,
#                 "dataframe_name": dataframe_name,
#                 "df": df #df.to_html(max_rows=1000, classes="panel-body table table-condensed", index=False, na_rep="", index_names=False, float_format=lambda x: round(x, 4)) if is_not_null(df) else None #, float_format=lambda x: "{}%".format(round(x*100, 2)))
#             }, context_instance=RequestContext(request))
#         else:
#             if is_not_null(df):
#                 csv = StringIO()
#                 df.to_csv(csv, encoding="utf8")
#                 csv.seek(0)
#                 csv = csv.getvalue()
#                 response = StreamingHttpResponse(csv, content_type="text/csv")
#                 response['Content-Disposition'] = 'attachment; filename="{}_{}_{}_{}.csv"'.format(project.name, dataframe_name, sample_name, dataframes.get_dataframe_time(dataframe_name, project, sample=sample))
#                 return response
#             else:
#                 return home(request)
#     else:
#         return home(request)