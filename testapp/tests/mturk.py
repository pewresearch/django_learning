# from __future__ import print_function
# import unittest
# import copy
# import os
# import pandas as pd
# import time
#
# from django.test import TestCase as DjangoTestCase
# from django.conf import settings
#
# from pewtils import is_not_null
#
# from django_learning.mturk import MTurk
# from django_learning.models import *
# from django_commander.commands import commands
#
# from testapp.models import MovieReview
#
#
# class MTurkTests(DjangoTestCase):
#
#     """
#     To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
#     """
#
#     def setUp(self):
#
#         reviews = pd.read_csv(os.path.join(settings.BASE_DIR, "testapp", "test_data.csv"))
#         for index, row in reviews[:50].iterrows():
#             if is_not_null(row['text']):
#                 obj = MovieReview.objects.create(text=row["text"][:200], id=index)
#
#     def test_mturk(self):
#
#         commands["django_learning_coding_mturk_clear_sandbox"]().run()
#
#         mturk = MTurk(sandbox=True)
#         mturk.clear_all_worker_blocks()
#
#         project = Project.objects.create(name="test_project", sandbox=True)
#         project.save()
#         commands["django_learning_coding_extract_sampling_frame"](
#             sampling_frame_name="all_documents"
#         ).run()
#         # commands["django_learning_coding_create_project"](
#         #     project_name="test_project",
#         #     sandbox=True
#         # ).run()
#         # TODO: figure out why this breaks
#         commands["django_learning_coding_extract_sample"](
#             project_name="test_project",
#             hit_type_name="test_hit_type",
#             sample_name="test_sample",
#             sampling_frame_name="all_documents",
#             size=10,
#             sandbox=True
#         ).run()
#         commands["django_learning_coding_create_sample_hits"](
#             project_name="test_project",
#             sample_name="test_sample",
#             num_coders=1,
#             sandbox=True
#         ).run()
#         commands["django_learning_coding_mturk_create_sample_hits"](
#             project_name="test_project",
#             sample_name="test_sample",
#             num_coders=1,
#             sandbox=True
#         ).run()
#         self.assertTrue(HIT.objects.count() == 20)
#         self.assertTrue(HIT.objects.filter(turk=True).count() == 10)
#         self.assertTrue(HIT.objects.filter(turk=False).count() == 10)
#
#         for qual_test in QualificationTest.objects.all():
#             print("Navigate to https://workersandbox.mturk.com/qualifications/{} to do the qualification test".format(
#                 QualificationTest.objects.values_list("turk_id", flat=True)[0])
#             )
#             import pdb
#             pdb.set_trace()
#             mturk.sync_qualification_test(qual_test)
#
#         print("Go to workersandbox.mturk.com, find the hits, and do a few, then press 'c' to continue")
#         print("If the hits don't appear, it's because the MTurk interface can be buggy.")
#         print("Check this out instead: https://github.com/jtjacques/mturk-manage")
#         print("Also you sometimes need to do more than one assignment for the sandbox API to return anything")
#         import pdb
#         pdb.set_trace()
#         time.sleep(30)
#
#         commands["django_learning_coding_mturk_sync_sample_hits"](
#             project_name="test_project",
#             sample_name="test_sample",
#             sandbox=True,
#             approve=True,
#             approve_probability=1.0,
#             update_blocks=True,
#             notify_blocks=True,
#             max_comp=0
#         ).run()
#         self.assertTrue(Assignment.objects.filter(turk_status="Approved").count()>0)
#         for q in Project.objects.get(name="test_project").questions.all():
#             self.assertTrue(Code.objects.filter(label__question=q).count() > 0)
#         HIT.objects.filter(turk_id__isnull=False).filter(assignments__isnull=False).update(turk_id=None)
#         time.sleep(5)
#         self.assertTrue(len(mturk.get_worker_blocks()["Maximum annual compensation"]) > 0)
#
#         commands["django_learning_coding_mturk_sync_sample_hits"](
#             project_name="test_project",
#             sample_name="test_sample",
#             sandbox=True,
#             approve=True,
#             approve_probability=1.0,
#             update_blocks=True,
#             notify_blocks=True,
#             max_comp=500
#         ).run()
#         time.sleep(5)
#         self.assertTrue(len(mturk.get_worker_blocks()["Maximum annual compensation"]) == 0)
#
#         commands["django_learning_coding_mturk_expire_sample_hits"](
#             project_name="test_project",
#             sample_name="test_sample",
#             sandbox=True
#         ).run()
#         self.assertTrue(HIT.objects.filter(turk_id__isnull=False).count() == 10)
#         commands["django_learning_coding_mturk_delete_sample_hits"](
#             project_name="test_project",
#             sample_name="test_sample",
#             sandbox=True
#         ).run()
#         Project.objects.get(name="test_project", sandbox=True).delete()
#         self.assertTrue(HIT.objects.count() == 0)
#         self.assertTrue(Assignment.objects.count() == 0)
#         self.assertTrue(Code.objects.filter(assignment__project__isnull=False).count() == 0)
#
#         commands["django_learning_coding_mturk_clear_sandbox"]().run()
#
# def tearDown(self):
#     pass