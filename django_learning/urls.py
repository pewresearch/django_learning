from django.conf.urls import url

from django_learning import views


app_name = 'django_learning'
urlpatterns = [
    url(r'^$', views.home, name="home"),
    url(r'^project/(?P<project_name>[\w\_]+)/extract_sample', views.extract_sample, name="extract_sample"),
    url(r'^project/(?P<project_name>[\w\_]+)/create_mturk_hits', views.create_sample_hits_mturk, name="create_sample_hits_mturk"),
    url(r'^project/(?P<project_name>[\w\_]+)/create_expert_hits', views.create_sample_hits_experts, name="create_sample_hits_experts"),
    url(r'^project/(?P<project_name>[\w\_]+)/edit_project_coders/(?P<mode>expert|mturk)', views.edit_project_coders, name="edit_project_coders"),
    url(r'^project/(?P<project_name>[\w\_]+)/sample/(?P<sample_name>[\w\_]+)/code_assignment/(?P<assignment_id>[0-9]+)$', views.code_assignment, name="code_specific_assignment"),
    url(r'^project/(?P<project_name>[\w\_]+)/sample/(?P<sample_name>[\w\_]+)/code_assignment$', views.code_assignment, name="code_assignment"),
    url(r'^project/(?P<project_name>[\w\_]+)/sample/(?P<sample_name>[\w\_]+)/complete_qualification/(?P<qualification_test_name>[\w\_]+)$', views.complete_qualification, name="complete_qualification"),
    url(r'^project/(?P<project_name>[\w\_]+)/sample/(?P<sample_name>[\w\_]+)/adjudicate_question/(?P<question_name>[\w\_]+)', views.adjudicate_question, name="adjudicate_question"),
    url(r'^project/(?P<project_name>[\w\_]+)/sample/(?P<sample_name>[\w\_]+)', views.view_sample, name="view_sample"),
    url(r'^project/(?P<project_name>[\w\_]+)', views.view_project, name="view_project"),
    url(r'^project/create/(?P<project_name>[\w\_]+)', views.create_project, name="create_project"),
    url(r'^sampling_frame/(?P<sampling_frame_name>[\w\_]+)/extract', views.extract_sampling_frame, name="extract_sampling_frame"),
    url(r'^sampling_frame/(?P<sampling_frame_name>[\w\_]+)', views.view_sampling_frame, name="view_sampling_frame"),
    url(r'^topic_model/(?P<model_id>[0-9]+)', views.edit_topic_model, name="edit_topic_model"),
    url(r'^topic_models', views.view_topic_models, name="view_topic_models"),

    #url(r'^get_dataframe/(?P<project_name>[\w\_]+)$', views.get_dataframe),
]