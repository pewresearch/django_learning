from django.conf.urls import url

from django_learning import views


app_name = 'django_learning'
urlpatterns = [
    url(r'^$', views.home, name="home"),
    url(r'^project/(?P<project_name>[\w\_]+)/extract_sample', views.extract_sample, name="extract_sample"),
    url(r'^project/(?P<project_name>[\w\_]+)/create_mturk_hits', views.create_sample_hits_mturk, name="create_sample_hits_mturk"),
    url(r'^project/(?P<project_name>[\w\_]+)/create_expert_hits', views.create_sample_hits_experts, name="create_sample_hits_experts"),
    url(r'^project/(?P<project_name>[\w\_]+)', views.view_project, name="view_project"),
    url(r'^create_project/(?P<project_name>[\w\_]+)', views.create_project, name="create_project"),
    url(r'^extract_sampling_frame/(?P<sampling_frame_name>[\w\_]+)', views.extract_sampling_frame, name="extract_sampling_frame"),
    url(r'^code_random_assignment/(?P<project_name>[\w\_]+)/(?P<sample_name>[\w\_]+)$', views.code_random_assignment, name="code_random_assignment"),
    url(r'^complete_qualification/(?P<project_name>[\w\_]+)/(?P<sample_name>[\w\_]+)/(?P<qualification_test_name>[\w\_]+)$', views.complete_qualification, name="complete_qualification"),
    #url(r'^get_dataframe/(?P<project_name>[\w\_]+)$', views.get_dataframe),
]