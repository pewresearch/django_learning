from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview


class CodingTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):

        now = datetime.date(2000, 1, 1)
        reviews = pd.read_csv(
            os.path.join(settings.BASE_DIR, "testapp", "test_data.csv")
        )
        for index, row in reviews.iterrows():
            if is_not_null(row["text"]):
                doc = Document.objects.create(
                    text=row["text"][:200],
                    id=index,
                    date=now + datetime.timedelta(days=index),
                )
                review = MovieReview.objects.create(document=doc, id=index)

    def test_coding(self):

        commands["django_learning_coding_extract_sampling_frame"](
            sampling_frame_name="all_documents"
        ).run()
        commands["django_learning_coding_create_project"](
            project_name="test_project", sandbox=True
        ).run()
        commands["django_learning_coding_extract_sample"](
            project_name="test_project",
            hit_type_name="test_hit_type",
            sample_name="test_sample",
            sampling_frame_name="all_documents",
            size=1000,
            sandbox=True,
            seed=42,
        ).run()
        commands["django_learning_coding_create_sample_hits"](
            project_name="test_project",
            sample_name="test_sample",
            num_coders=2,
            sandbox=True,
        ).run()
        commands["django_learning_coding_extract_sample"](
            project_name="test_project",
            hit_type_name="test_hit_type",
            sample_name="test_sample_holdout",
            sampling_frame_name="all_documents",
            size=100,
            sandbox=True,
            seed=42,
        ).run()
        commands["django_learning_coding_create_sample_hits"](
            project_name="test_project",
            sample_name="test_sample_holdout",
            num_coders=2,
            sandbox=True,
        ).run()

        coder1 = Coder.objects.create(name="coder1")
        coder2 = Coder.objects.create(name="coder2")
        test_project = Project.objects.get(name="test_project")
        test_project.coders.add(coder1)
        test_project.coders.add(coder2)

        for sample_name in ["test_sample", "test_sample_holdout"]:
            df = Document.objects.filter(samples__name=sample_name).dataframe(
                "document_text", refresh=True
            )
            df["is_good"] = df["text"].str.contains(r"good|great|excellent").astype(int)
            df = df.sort_values("pk")
            random_seed = 42
            for question in ["test_checkbox", "test_radio"]:
                coder1_docs = df[df["is_good"] == 1].sample(
                    frac=0.8, random_state=random_seed
                )
                coder2_docs = df[df["is_good"] == 1].sample(
                    frac=0.8, random_state=random_seed + 1
                )
                df["coder1"] = df["pk"].map(
                    lambda x: 1 if x in coder1_docs["pk"].values else 0
                )
                df["coder2"] = df["pk"].map(
                    lambda x: 1 if x in coder2_docs["pk"].values else 0
                )
                label1 = Label.objects.filter(question__name=question).get(value="1")
                label0 = Label.objects.filter(question__name=question).get(value="0")
                for index, row in df.iterrows():
                    su = SampleUnit.objects.filter(sample__name=sample_name).get(
                        document_id=row["pk"]
                    )
                    hit = HIT.objects.get(sample_unit=su)
                    for coder, coder_name in [(coder1, "coder1"), (coder2, "coder2")]:
                        assignment = Assignment.objects.create(hit=hit, coder=coder)
                        Code.objects.create(
                            label=label1 if row[coder_name] else label0,
                            assignment=assignment,
                        )
                random_seed += 42

        from django_learning.utils.dataset_extractors import dataset_extractors

        extractor = dataset_extractors["document_coder_label_dataset"](
            project_name="test_project",
            sample_names=["test_sample"],
            question_names=["test_checkbox"],
            coder_filters=[],
            document_filters=[],
            ignore_stratification_weights=True,
            sandbox=True,
        )
        scores = extractor.compute_scores(
            refresh=True, min_overlap=5, discrete_classes=True
        )
        self.assertAlmostEqual(scores["cohens_kappa"].mean(), 0.78978, 4)
        scores = extractor.compute_overall_scores()
        self.assertAlmostEqual(scores["alpha"], 0.789885, 4)
        self.assertAlmostEqual(scores["fleiss_kappa"], 0.78978, 4)

        df = extractor.extract()
        self.assertEqual(len(df), 2000)
        self.assertEqual(df["document_id"].nunique(), 1000)
        self.assertEqual(df["coder_id"].nunique(), 2)

        # extractor = dataset_extractors["document_coder_dataset"](
        #     project_name="test_project",
        #     sample_names=["test_sample"],
        #     question_names=["test_checkbox"],
        #     coder_filters=[],
        #     document_filters=[],
        #     ignore_stratification_weights=True,
        #     sandbox=True,
        # )

        params = {
            "project_name": "test_project",
            "sample_names": ["test_sample"],
            "coder_filters": [],
            "document_filters": [],
            "ignore_stratification_weights": True,
            "sandbox": True,
        }
        label_pos = Label.objects.filter(question__name="test_checkbox").get(value="1")
        for update_params, values in [
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": False,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                (901, 32, 67, 67, 32, 901),
            ),
            (
                {
                    "coder_aggregation_function": "max",
                    "convert_to_discrete": False,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                (901, 0, 99, 67, 0, 933),
            ),
            (
                {
                    "coder_aggregation_function": "min",
                    "convert_to_discrete": False,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                (933, 0, 67, 99, 0, 901),
            ),
            (
                {
                    "coder_aggregation_function": "median",
                    "convert_to_discrete": False,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                (901, 32, 67, 67, 32, 901),
            ),
        ]:
            params.update(update_params)
            extractor = dataset_extractors["document_dataset"](**params)
            df = extractor.extract()
            if len(extractor.outcome_columns) == 2:
                self.assertEqual(len(df[df["label_11"] == 0.0]), values[0])
                self.assertEqual(len(df[df["label_11"] == 0.5]), values[1])
                self.assertEqual(len(df[df["label_11"] == 1.0]), values[2])
                self.assertEqual(len(df[df["label_12"] == 0.0]), values[3])
                self.assertEqual(len(df[df["label_12"] == 0.5]), values[4])
                self.assertEqual(len(df[df["label_12"] == 1.0]), values[5])

        for update_params, values in [
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 1.0,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                {"12": 901, "11": 67, "None": 32},
            ),
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 0.0,
                    "base_class_id": None,
                    "question_names": ["test_checkbox"],
                },
                {"12": 901, "11": 99},
            ),
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 0.5,
                    "base_class_id": label_pos.pk,
                    "question_names": ["test_checkbox"],
                },
                {"12": 933, "11": 67},
            ),
            (
                {
                    "coder_aggregation_function": "mean",
                    "convert_to_discrete": True,
                    "threshold": 0.4,
                    "base_class_id": None,
                    "question_names": ["test_checkbox", "test_radio"],
                },
                {"0101": 905, "1001": 25, "0110": 30, "1010": 40},
            ),
        ]:
            params.update(update_params)
            extractor = dataset_extractors["document_dataset"](**params)
            df = extractor.extract()
            vals = df["label_id"].value_counts().to_dict()
            self.assertEqual(vals, values)

        # commands["django_learning_models_train_document_classifier"](name="test_model", pipeline_name="test", refresh_dataset=True, refresh_model=True).run()
        frame = SamplingFrame.objects.get(name="all_documents")

        ### Simple classifier
        model = DocumentClassificationModel.objects.create(
            name="test_model", pipeline_name="test", sampling_frame=frame
        )
        model.extract_dataset(refresh=True)
        model.load_model(refresh=True, num_cores=1)
        model.describe_model()
        model.get_cv_prediction_results(refresh=True)
        model.get_test_prediction_results(refresh=True)
        model.find_probability_threshold(save=True)
        test_scores = model.get_test_prediction_results()
        fold_scores = model.get_cv_prediction_results()

        self.assertEqual(len(model.dataset), 1000)
        self.assertEqual(len(model.train_dataset), 750)
        self.assertEqual(len(model.test_dataset), 250)
        self.assertEqual(len(model.cv_folds), 5)
        for a, b in model.cv_folds:
            self.assertGreaterEqual(len(a), 599)
            self.assertLessEqual(len(a), 601)
            self.assertGreaterEqual(len(b), 149)
            self.assertLessEqual(len(b), 151)
        self.assertEqual(list(set(test_scores["n"].values))[0], 250)
        self.assertEqual(list(set(fold_scores["n"].values))[0], 150)

        for label_id, metric, expected_val in [
            (11, "precision", 0.96),
            (11, "recall", 0.82),
            (11, "n", 250),
            (11, "matthews_corrcoef", 0.87),
            (11, "cohens_kappa", 0.87),
            (11, "accuracy", 0.972),
            (12, "precision", 0.97),
            (12, "recall", 1.0),
            (12, "n", 250),
            (12, "matthews_corrcoef", 0.87),
            (12, "cohens_kappa", 0.87),
            (12, "accuracy", 0.97),
        ]:
            val = test_scores.loc[
                test_scores["outcome_column"] == "label_id__{}".format(label_id)
            ][metric].values[0]
            self.assertAlmostEqual(val, expected_val, 2)
            # print("{}: {} ({} expected".format(metric, val, expected_val))

        model = DocumentClassificationModel.objects.create_or_update(
            {"name": "test_model"}, {"pipeline_name": "test", "sampling_frame": frame}
        )
        model.extract_dataset(refresh=False)
        model.load_model(refresh=False)

        ### Classifier with holdout
        model = DocumentClassificationModel.objects.create(
            name="test_model_with_holdout",
            pipeline_name="test_with_holdout",
            sampling_frame=frame,
        )
        model.extract_dataset(refresh=True)
        model.load_model(refresh=True, num_cores=2)
        model.describe_model()
        model.get_test_prediction_results(refresh=True)
        model.find_probability_threshold(save=True)
        test_scores = model.get_test_prediction_results()

        self.assertEqual(len(model.dataset), 1100)
        self.assertEqual(len(model.train_dataset), 1000)
        self.assertEqual(len(model.test_dataset), 100)
        self.assertEqual(list(set(test_scores["n"].values))[0], 100)

        for label_id, metric, expected_val in [
            (11, "precision", 1.0),
            (11, "recall", 0.82),
            (11, "n", 100),
            (11, "matthews_corrcoef", 0.89),
            (11, "cohens_kappa", 0.89),
            (11, "accuracy", 0.98),
            (12, "precision", 0.98),
            (12, "recall", 1.0),
            (12, "n", 100),
            (12, "matthews_corrcoef", 0.89),
            (12, "cohens_kappa", 0.89),
            (12, "accuracy", 0.98),
        ]:
            val = test_scores.loc[
                test_scores["outcome_column"] == "label_id__{}".format(label_id)
            ][metric].values[0]
            # print("{}: {} ({} expected".format(metric, val, expected_val))
            self.assertAlmostEqual(val, expected_val, 2)

        # TODO: test keyword oversampling and weighting

    def tearDown(self):

        from django.conf import settings
        import shutil

        shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
