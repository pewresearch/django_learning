from pewtils.django import get_model


def filter(self, df, model_name, filter_value):

    from django_learning.utils.dataset_extractors import dataset_extractors
    other_model = get_model("DocumentClassificationModel", app_name="django_learning").objects.get(name=model_name)
    predicted_df = dataset_extractors["model_prediction_dataset"](dataset=df, learning_model=other_model).extract(refresh=True)
    doc_ids = predicted_df[predicted_df[other_model.dataset_extractor.outcome_column]==filter_value]['document_id'].unique()
    df = df[df['document_id'].isin(doc_ids)]

    return df


