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

from django_learning.models import *
from django_learning.functions import *
from django_learning.utils.projects import projects as project_configs
from django_learning.utils.sampling_frames import sampling_frames as sampling_frame_configs
from django_learning.utils.sampling_methods import sampling_methods as sampling_method_configs
from django_learning.utils.project_hit_types import project_hit_types as project_hit_type_configs

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
        "hit_types": project_hit_type_configs, # TODO: shouldn't this be only HITTypes associated with the project?
        "sampling_methods": sampling_method_configs,
        "sampling_frames": SamplingFrame.objects.all()
        # "dataframes": dataframes.DATAFRAME_NAMES
    })


@login_required
def create_project(request, project_name):

    run_command_task.delay("create_project", {
        "project_name": project_name
    })

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

    return render(request, "django_learning/sample.html", {
        "sample": sample,
        "total_expert_hits": total_expert_hits,
        "available_expert_hits": available_expert_hits,
        "completed_expert_hits": completed_expert_hits,
        "total_turk_hits": total_turk_hits,
        "completed_turk_hits": completed_turk_hits,
        "total_turk_assignments": total_turk_assignments,
        "completed_turk_assignments": completed_turk_assignments
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
            return code_random_assignment(request, project.name, sample.name, skip_post=True)
        else:
            QualificationAssignment.objects.create_or_update({"test": qual_test, "coder": request.user.coder}, return_object=False)
            return _render_qualification_test(request, project, sample, qual_test)

    else:

        return view_project(request, project.name)


@login_required
def code_random_assignment(request, project_name, sample_name, skip_post=False):

    if request.method == "POST" and not skip_post:

        hit = HIT.objects.get(pk=request.POST.get("hit_id"))
        if hit.sample and request.user.coder in hit.sample.project.coders.all():

            assignment = Assignment.objects.get(hit=hit, coder=request.user.coder)
            if not assignment.time_finished:

                assignment.time_finished = datetime.datetime.now()
                assignment.save()

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

            else:

                return view_project(request, project_name)

    project = Project.objects.get(name=project_name)

    if request.user.coder in project.coders.all():

        sample = Sample.objects.get(project=project, name=sample_name)
        request.user.coder._clear_abandoned_sample_assignments(sample)

        qual_tests = project.qualification_tests.all() | sample.hit_type.qualification_tests.all()
        for qual_test in qual_tests:
            try: qual = QualificationAssignment.objects.filter(time_finished__isnull=False).get(test=qual_test, coder=request.user.coder)
            except QualificationAssignment.DoesNotExist: qual = None
            if not qual:
                return complete_qualification(request, project.name, sample.name, qual_test.name)

        if sample.hit_type.is_qualified(request.user.coder):

            hits_available = filter_hits(sample=sample, unfinished_only=True, experts_only=True, exclude_coders=[request.user.coder])\
                .annotate(c=Count("assignments"))\
                .order_by("-c")\
                .distinct()

            if hits_available.count() > 0:

                hit = hits_available[0]
                Assignment.objects.create_or_update({"hit": hit, "coder": request.user.coder}, return_object=False)
                return _render_hit(request, project, sample, hit, remaining_count=len(hits_available))

            else:
                return view_project(request, project_name)
        else:
            return view_project(request, project_name)
    else:
        return view_project(request, project_name)


def _render_qualification_test(request, project, sample, qual_test):

    return render(request, "django_learning/qual_test.html", {
        "project": project,
        "sample": sample,
        "qual_test": qual_test
    })


def _render_hit(request, project, sample, hit, remaining_count=None):

    if hit.template_name:
        template = "{}.html".format(hit.template_name)
        # for folder in settings.DJANGO_LEARNING_HIT_TEMPLATE_DIRS:
        #     path = os.path.join(folder, "{}.html".format(hit.template_name))
        #     if os.path.exists(path):
        #         template = path
        # # template = "custom_hits/{}.html".format(hit.template_name)
    else:
        template = "django_learning/hit.html"

    return render(request, template, {
        "remaining_count": remaining_count,
        "project": project,
        "sample": sample,
        "hit": hit
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