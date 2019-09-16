from __future__ import print_function
import unittest
import copy
import os
import pandas as pd
import time

from django.test import TestCase as DjangoTestCase
from django.conf import settings

from pewtils import is_not_null

from django_learning.mturk import MTurk
from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview


class CodingTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):

        reviews = pd.read_csv(os.path.join(settings.BASE_DIR, "testapp", "test_data.csv"))
        for index, row in reviews.iterrows():
            if is_not_null(row['text']):
                doc = Document.objects.create(text=row["text"][:200], id=index)
                review = MovieReview.objects.create(document=doc, id=index)

    def test_coding(self):

        commands["django_learning_coding_extract_sampling_frame"](
            sampling_frame_name="all_documents"
        ).run()
        commands["django_learning_coding_create_project"](
            project_name="test_project",
            sandbox=True
        ).run()
        commands["django_learning_coding_extract_sample"](
            project_name="test_project",
            hit_type_name="test_hit_type",
            sample_name="test_sample",
            sampling_frame_name="all_documents",
            size=100,
            sandbox=True
        ).run()
        commands["django_learning_coding_create_sample_hits"](
            project_name="test_project",
            sample_name="test_sample",
            num_coders=3,
            sandbox=True
        ).run()

        df = Document.objects.filter(samples__name="test_sample").dataframe("document_text", refresh=True)
        df['is_good'] = df['text'].str.contains(r"good|great|excellent").astype(int)
        coder1 = Coder.objects.create(name="coder1")
        coder2 = Coder.objects.create(name="coder2")
        test_project = Project.objects.get(name="test_project")
        test_project.coders.add(coder1)
        test_project.coders.add(coder2)
        coder1_docs = df[df['is_good']==1].sample(frac=.8)
        coder2_docs = df[df['is_good']==1].sample(frac=.8)
        df['coder1'] = df['pk'].map(lambda x: 1 if x in coder1_docs['pk'].values else 0)
        df['coder2'] = df['pk'].map(lambda x: 1 if x in coder2_docs['pk'].values else 0)
        label1 = Label.objects.filter(question__name="test_checkbox").get(value="1")
        label0 = Label.objects.filter(question__name="test_checkbox").get(value="0")
        for index, row in df.iterrows():
            su = SampleUnit.objects.filter(sample__name="test_sample").get(document_id=row['pk'])
            hit = HIT.objects.get(sample_unit=su)
            for coder, coder_name in [
                (coder1, "coder1"),
                (coder2, "coder2")
            ]:
                assignment = Assignment.objects.create(
                    hit=hit,
                    coder=coder
                )
                Code.objects.create(
                    label=label1 if row[coder_name] else label0,
                    assignment=assignment
                )

        from django_learning.utils.dataset_extractors import dataset_extractors
        extractor = dataset_extractors["document_coder_label_dataset"](
            project_name="test_project",
            sample_names=["test_sample"],
            question_names=["test_checkbox"],
            coder_filters=[],
            document_filters=[],
            ignore_stratification_weights=True,
            sandbox=True
        )
        scores = extractor.compute_scores(refresh=True, min_overlap=5, discrete_classes=True)
        import pdb; pdb.set_trace()
        self.assertAlmostEqual(scores['cohens_kappa'].mean(), .7558)

        # self.assertTrue(HIT.objects.count() == 10)
        # self.assertTrue(HIT.objects.filter(turk=False).count() == 10)
        #
        #
        # self.assertTrue(Assignment.objects.filter(turk_status="Approved").count()>0)
        # for q in Project.objects.get(name="test_project").questions.all():
        #     self.assertTrue(Code.objects.filter(label__question=q).count() > 0)
        # HIT.objects.filter(turk_id__isnull=False).filter(assignments__isnull=False).update(turk_id=None)
        # time.sleep(5)
        # self.assertTrue(len(mturk.get_worker_blocks()["Maximum annual compensation"]) > 0)
        #
        # commands["django_learning_coding_mturk_sync_sample_hits"](
        #     project_name="test_project",
        #     sample_name="test_sample",
        #     sandbox=True,
        #     approve=True,
        #     approve_probability=1.0,
        #     update_blocks=True,
        #     notify_blocks=True,
        #     max_comp=500
        # ).run()
        # time.sleep(5)
        # self.assertTrue(len(mturk.get_worker_blocks()["Maximum annual compensation"]) == 0)
        #
        # commands["django_learning_coding_mturk_expire_sample_hits"](
        #     project_name="test_project",
        #     sample_name="test_sample",
        #     sandbox=True
        # ).run()
        # self.assertTrue(HIT.objects.filter(turk_id__isnull=False).count() == 10)
        # commands["django_learning_coding_mturk_delete_sample_hits"](
        #     project_name="test_project",
        #     sample_name="test_sample",
        #     sandbox=True
        # ).run()
        # Project.objects.get(name="test_project", sandbox=True).delete()
        # self.assertTrue(HIT.objects.count() == 0)
        # self.assertTrue(Assignment.objects.count() == 0)
        # self.assertTrue(Code.objects.filter(assignment__project__isnull=False).count() == 0)
        #
        # commands["django_learning_coding_mturk_clear_sandbox"]().run()

    def tearDown(self):
        pass