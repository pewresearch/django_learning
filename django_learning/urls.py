from django.urls import re_path
from django_learning import views


app_name = "django_learning"
urlpatterns = [
    re_path(r"^$", views.home, name="home"),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/extract_sample",
        views.extract_sample,
        name="extract_sample",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/create_mturk_hits",
        views.create_sample_hits_mturk,
        name="create_sample_hits_mturk",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/create_expert_hits",
        views.create_sample_hits_experts,
        name="create_sample_hits_experts",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/edit_project_coders/(?P<mode>expert|mturk)",
        views.edit_project_coders,
        name="edit_project_coders",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/sample/(?P<sample_name>[\w\_\-]+)/code_assignment/(?P<assignment_id>[0-9]+)$",
        views.code_assignment,
        name="code_specific_assignment",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/sample/(?P<sample_name>[\w\_\-]+)/code_assignment$",
        views.code_assignment,
        name="code_assignment",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/sample/(?P<sample_name>[\w\_\-]+)/expert_assignments$",
        views.view_expert_assignments,
        name="view_expert_assignments",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/sample/(?P<sample_name>[\w\_\-]+)/complete_qualification/(?P<qualification_test_name>[\w\_\-]+)$",
        views.complete_qualification,
        name="complete_qualification",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/sample/(?P<sample_name>[\w\_\-]+)/adjudicate_question/(?P<question_name>[\w\_\-]+)",
        views.adjudicate_question,
        name="adjudicate_question",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/sample/download/(?P<sample_name>[\w\_\-]+)",
        views.download_sample,
        name="download_sample",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)/sample/(?P<sample_name>[\w\_\-]+)",
        views.view_sample,
        name="view_sample",
    ),
    re_path(
        r"^project/(?P<project_name>[\w\_\-]+)", views.view_project, name="view_project"
    ),
    re_path(
        r"^project/create/(?P<project_name>[\w\_\-]+)",
        views.create_project,
        name="create_project",
    ),
    re_path(
        r"^sampling_frame/(?P<sampling_frame_name>[\w\_\-]+)/extract",
        views.extract_sampling_frame,
        name="extract_sampling_frame",
    ),
    re_path(
        r"^sampling_frame/(?P<sampling_frame_name>[\w\_\-]+)",
        views.view_sampling_frame,
        name="view_sampling_frame",
    ),
    re_path(
        r"^topic_model/(?P<model_id>[0-9]+)",
        views.edit_topic_model,
        name="edit_topic_model",
    ),
    re_path(r"^topic_models", views.view_topic_models, name="view_topic_models"),
    re_path(
        r"^document_classification_model/(?P<model_name>[\w\_\-]+)",
        views.view_document_classification_model,
        name="view_document_classification_model",
    ),
    re_path(
        r"^document_classifications/(?P<model_name>[\w\_\-]+)/(?P<label_id>[0-9]+)",
        views.view_document_classifications,
        name="view_document_classifications",
    ),
    # re_path(r'^get_dataframe/(?P<project_name>[\w\_]+)$', views.get_dataframe),
]
