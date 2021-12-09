# Machine learning

## Components

#### Balancing Variables

Balancing variables can be used to specify additional partitions in your data that should be used in weighting.
When extracting a `document_dataset`, `document_coder_datset`, or `document_coder_label` dataset, you can
pass in a keyword argument containing the names of `balancing_variables` that correspond to functions defined
in your `settings.DJANGO_LEARNING_BALANCING_VARIABLES` folders. Weights will be computed to evenly balance all
classes within the partitions (using the combinations if multiple variables are included) and the results will
be contained in a `balancing_weight` column in the returned dataset. If you specify a balancing weight in the
training dataset for a machine learning pipeline, the `LearningModel` will use the balancing weight from the
dataset when computing its training weight. If class weights and/or sampling weights are included, these will
be multiplied by the balancing weight. Balancing weights will NOT be used when evaluating model performance.
They exist in Django Learning solely for the training phase of machine learning, or simply for your own
convenience when you're extracting datasets.

Example of a balancing variable file that balances documents evenly by month:

```python
from django_pewtils import get_model

def var_mapper(x):

    doc = get_model("Document", app_name="django_learning").objects.get(
        pk=x["document_id"]
    )
    if doc.date and doc.date.month and doc.date.year:
        return "{}_{}".format(doc.date.year, doc.date.month)
    else:
        return None
```

- Feature Extractors

All feature extractors extend traditional Scikit-Learn processors and expect to receive a dataframe that includes
a `document_id` and `text` column. This allows for more sophisticated caching and feature extraction.
- Models
- Pipelines
- Preprocessors
- Regex Replacers
- Scoring Functions
- Stopword Sets
- Stopword Whitelists=

#### Using a separate test/hold-out dataset for evaluation

#### Dealing with dependencies

Django Learning doesn't automatically impose dependencies because you may want to filter based on A)
human labels (which might require a collapse rule) or B) the decisions of a model trained to predict
the dependency.  Django Learning provides several Dataset Document Filters to specify this, respectively:

1) filter_by_existing_code
2) filter_by_other_model_dataset
3) filter_by_other_model_prediction