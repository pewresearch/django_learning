import datetime
import os

import pandas as pd

from django.conf import settings

from pewtils import is_not_null, recursive_update
from django_commander.commands import commands
from django_learning.models import *

from testapp.models import MovieReview


def set_up_test_project(limit=None):

    if Project.objects.filter(name="test_project").count() == 0:
        now = datetime.date(2000, 1, 1)
        reviews = pd.read_csv(
            os.path.join(settings.BASE_DIR, "testapp", "test_data.csv")
        )
        count = 0
        for index, row in reviews.iterrows():
            if is_not_null(row["text"]):
                doc = Document.objects.create(
                    text=row["text"][:200],
                    id=index,
                    date=now + datetime.timedelta(days=index),
                )
                review = MovieReview.objects.create(document=doc, id=index)
                count += 1
                if limit and count >= limit:
                    break

        commands["django_learning_coding_extract_sampling_frame"](
            sampling_frame_name="all_documents"
        ).run()
        commands["django_learning_coding_create_project"](
            project_name="test_project"
        ).run()


def set_up_test_sample(sample_name, size):

    set_up_test_project(limit=size + 50)

    if sample_name == "test_sample":
        method = "random"
    elif sample_name == "test_sample_holdout":
        method = "random"
    elif sample_name == "test_sample_keyword_oversample":
        method = "keyword_oversample"

    commands["django_learning_coding_extract_sample"](
        project_name="test_project",
        sample_name=sample_name,
        sampling_frame_name="all_documents",
        sampling_method=method,
        size=size,
        seed=42,
    ).run()
    commands["django_learning_coding_create_sample_hits"](
        project_name="test_project",
        sample_name=sample_name,
        hit_type_name="test_hit_type",
        num_coders=2,
    ).run()

    coder1 = Coder.objects.create_or_update({"name": "coder1"})
    coder2 = Coder.objects.create_or_update({"name": "coder2"})
    test_project = Project.objects.get(name="test_project")
    test_project.coders.add(coder1)
    test_project.coders.add(coder2)

    df = pd.DataFrame.from_records(
        Document.objects.filter(samples__name=sample_name).values("pk", "text")
    )

    df = df.sort_values("pk")

    question = "test_checkbox"
    random_seed = 42
    df["is_good"] = df["text"].str.contains(r"good|great|excellent").astype(int)
    coder1_docs = df[df["is_good"] == 1].sample(frac=0.8, random_state=random_seed)
    coder2_docs = df[df["is_good"] == 1].sample(frac=0.8, random_state=random_seed + 1)
    df["coder1"] = df["pk"].map(lambda x: 1 if x in coder1_docs["pk"].values else 0)
    df["coder2"] = df["pk"].map(lambda x: 1 if x in coder2_docs["pk"].values else 0)
    label1 = Label.objects.filter(question__name=question).get(value="1")
    label0 = Label.objects.filter(question__name=question).get(value="0")
    for index, row in df.iterrows():
        su = SampleUnit.objects.filter(sample__name=sample_name).get(
            document_id=row["pk"]
        )
        hit = HIT.objects.get(sample_unit=su)
        for coder, coder_name in [(coder1, "coder1"), (coder2, "coder2")]:
            assignment, _ = Assignment.objects.get_or_create(hit=hit, coder=coder)
            Code.objects.create(
                label=label1 if row[coder_name] else label0, assignment=assignment
            )
            assignment.time_finished = datetime.datetime.now()
            assignment.save()
            hit.save()

    question = "test_radio"
    random_seed += 42
    df["genre"] = "other"
    df.loc[df["text"].str.contains(r"comedy|funny|laugh"), "genre"] = "comedy"
    df.loc[df["text"].str.contains(r"action|suspense|adventur"), "genre"] = "action"
    df.loc[
        df[df["genre"] == "comedy"].sample(frac=0.05, random_state=random_seed).index,
        "genre",
    ] = "other"
    df.loc[
        df[df["genre"] == "action"]
        .sample(frac=0.05, random_state=random_seed + 1)
        .index,
        "genre",
    ] = "other"
    for index, row in df.iterrows():
        su = SampleUnit.objects.filter(sample__name=sample_name).get(
            document_id=row["pk"]
        )
        hit = HIT.objects.get(sample_unit=su)
        assignment, _ = Assignment.objects.get_or_create(hit=hit, coder=coder1)
        Code.objects.create(
            label=Label.objects.filter(question__name=question).get(value=row["genre"]),
            assignment=assignment,
        )
        assignment.time_finished = datetime.datetime.now()
        assignment.save()
        hit.save()


def get_base_dataset_parameters(
    extractor_name,
    sample_name="test_sample",
    question_name="test_checkbox",
    params=None,
):

    base_params = {
        "project_name": "test_project",
        "sample_names": [sample_name],
        "question_names": [question_name],
        "document_filters": [],
        "coder_filters": [],
        "balancing_variables": [],
        "ignore_stratification_weights": False,
    }
    if extractor_name == "document_dataset":
        try:
            base_class_id = (
                get_model("Question", app_name="django_learning")
                .objects.filter(project__name="test_project")
                .get(name=question_name)
                .labels.get(value="0")
                .pk
            )
            base_params["base_class_id"] = base_class_id
        except Label.DoesNotExist:
            pass
        base_params["threshold"] = 0.4
        base_params["convert_to_discrete"] = True

    if is_not_null(params):
        params = recursive_update(base_params, params)
    else:
        params = base_params

    return params


def extract_dataset(
    extractor_name,
    sample_name="test_sample",
    question_name="test_checkbox",
    params=None,
):

    from django_learning.utils.dataset_extractors import dataset_extractors

    params = get_base_dataset_parameters(
        extractor_name,
        sample_name=sample_name,
        question_name=question_name,
        params=params,
    )
    extractor = dataset_extractors[extractor_name](**params)
    df = extractor.extract(refresh=True)

    return df


def get_test_model(pipeline_name, run=True):

    frame = SamplingFrame.objects.get(name="all_documents")
    model = DocumentClassificationModel.objects.create(
        name=pipeline_name, pipeline_name=pipeline_name, sampling_frame=frame
    )
    if run:
        model.extract_dataset(refresh=True)
        model.load_model(refresh=True, num_cores=1)
        model.describe_model()
        if model.parameters["model"]["cv"] > 1:
            model.get_cv_prediction_results(refresh=True)
        model.get_test_prediction_results(refresh=True)
        model.find_probability_threshold(save=True)

    return model


def create_qualification_test():

    test, _ = QualificationTest.objects.get_or_create(name="test_qualification")

    for coder_name in ["coder1", "coder2"]:
        coder, _ = Coder.objects.get_or_create(name=coder_name)
        assignment = QualificationAssignment.objects.create_or_update(
            {"test": test, "coder": coder}
        )
        assignment.time_finished = datetime.datetime.now()
        assignment.save()
        question = test.questions.get(name="q1")
        if coder_name == "coder1":
            label_id = question.labels.get(value="1").pk
        else:
            label_id = question.labels.get(value="0").pk
        question.update_assignment_response(assignment, label_id)
