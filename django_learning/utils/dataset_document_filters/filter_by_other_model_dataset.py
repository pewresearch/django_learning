from pewtils.django import get_model


def filter(self, df, model_name, filter_value):

    other_model = get_model("DocumentClassificationModel", app_name="django_learning").objects.get(name=model_name)
    other_model.extract_dataset()
    other_df = other_model.dataset
    doc_ids = other_df[other_df[other_model.dataset_extractor.outcome_column]==filter_value]['document_id'].unique()
    df = df[df['document_id'].isin(doc_ids)]

    return df


