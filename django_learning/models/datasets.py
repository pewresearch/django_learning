

class Dataset(LoggedExtendedModel):

    name
    outcome_column
    outcome_columns
    cache_hash

    extractor_name

    index_levels
    standardize_coders
    convert_to_discrete
    threshold
    base_class_id
    discrete_classes
    valid_label_ids



# don't use a model for this
# in the model pipelines, replace "codes" with "extractor"
# which should have a name, and outcome_column, and then all of the required kwargs
# use the name to import from the dataset_extractors dictionary
# custom extractors can be put in the project-specific folder and accessed the same way