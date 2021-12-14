Extracting datasets and collapsing coders
==========================================

## Components
### Dataset Code Filters
Code filters are the broadest way of excluding codes from a dataset when pulling an extract from the database.
You can pass a parameter called `code_filters` to a `document_dataset`, `document_coder_datset`, or
`document_coder_label` extractor, which should be a list of tuples of the form `(FILTER_NAME, args, kwargs)`,
where `FILTER_NAME` corresponds to a file with a `filter` function defined in it, found in a folder contained
in one of your `settings.DJANGO_LEARNING_DATASET_CODER_FILTERS` folders. The function should appear as follows:

```python
def filter(self, df, *args, **kwargs):

    return df[df["label_id"] == 12]
```
The function will receive the full dataframe of codes as the first argument (`df`), followed by any additional
arguments and keyword arguments that you specified in a list and dictionary, respectively, when passing the name
of the filter file to the extractor.

NOTE: code filters are always executed first, followed by coder filters, and then document filters.


### Dataset Coder Filters
Function the same way as code filters

### Dataset Document Filters
Function the same way as code/coder filters.

WITH ONE EXCEPTION: they will be also applied to the sampling frame when computing sampling weights.
If you want to extract a dataset that's filtered in some way, and then weight it back to the full unfiltered sampling
frame, you'll need to do that manually. Generally speaking, if you're systematically excluding certain observations
from a sample, the subset shouldn't be used to make inferences about any data that was excluded. Right now, Django
Learning assumes that if you're filtering to documents pertaining category or range (like dates), then those filters
should be applied whenever the dataset is related back to the broader population from which it was drawn. This is
particularly relevant when using a dataset to train a machine learning model; document filters will be propagated and
used not only to compute sampling weights, but they will ALSO be automatically applied a trained model is applied to a
dataset. Document filters are considered to be a universal scoping mechanism and they move in one direction only.

### Dataset Extractors

