Computing inter-rater reliability
==========================================

Computing full-codebook IRR
----------------------------

Let's assume that we have a project with two questions. To calculate IRR across the full codebook, with
every unique combination of codes treated as a different value, we can first extract a dataset like so:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    extractor = dataset_extractors["document_coder_label_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random"],
        "question_names": ["review_sentiment", "another_question"],
    })
    dataset = extractor.extract(refresh=True)
    >>> dataset

    document_id  coder_id coder_name  coder_is_mturk  sampling_weight       date label_id                                               text document_type
              0         3     coder1           False              1.0 2000-01-01    00101  plot : two teen couples go to a church party ,...  movie_review
              0         4     coder2           False              1.0 2000-01-01    00001  plot : two teen couples go to a church party ,...  movie_review
              4         3     coder1           False              1.0 2000-01-05    00101  synopsis : a mentally unstable man undergoing ...  movie_review
              4         4     coder2           False              1.0 2000-01-05    00001  synopsis : a mentally unstable man undergoing ...  movie_review
              5         3     coder1           False              1.0 2000-01-06    00101  capsule : in 2176 on the planet mars police ta...  movie_review

We can then compute the IRR scores using the ``compute_scores_from_dataset`` function. This will return a dataframe of
various agreement metrics for each distinct code combination value:

.. code:: python

    from django_learning.utils.scoring import compute_scores_from_dataset
    scores = compute_scores_from_dataset(
        dataset,
        "document_id",
        "label_id",
        "coder_id",
        weight_column="sampling_weight",
        discrete_classes=True,
        pos_label=None,
    )
    >>> scores

    coder1  coder2    n   outcome_column  pos_label  coder1_mean  coder1_std  coder1_mean_unweighted  ...        f1  precision    recall  precision_recall_min  matthews_corrcoef   roc_auc  pct_agree_unweighted  cohens_kappa
         1       2  200         label_id        NaN     1176.995  214.623431                1176.995  ...  0.970000   0.970000  0.970000              0.970000           0.911203       NaN                  0.97           NaN
         1       2  200  label_id__00101        1.0        0.805    0.028086                   0.805  ...  0.987578   0.987578  0.987578              0.987578           0.936296  0.968148                  0.98      0.936296
         1       2  200  label_id__00110        1.0        0.050    0.015450                   0.050  ...  0.800000   0.800000  0.800000              0.800000           0.789474  0.894737                  0.98      0.789474
         1       2  200  label_id__01001        1.0        0.040    0.013891                   0.040  ...  1.000000   1.000000  1.000000              1.000000           1.000000  1.000000                  1.00      1.000000
         1       2  200  label_id__10001        1.0        0.100    0.021266                   0.100  ...  0.950000   0.950000  0.950000              0.950000           0.944444  0.972222                  0.99      0.944444
         1       2  200  label_id__10010        1.0        0.005    0.005000                   0.005  ...  0.000000   0.000000  0.000000              0.000000          -0.005025  0.497487                  0.99     -0.005025

If you have more than two coders in the dataset, the dataframe will return scores for all pairwise combinations of
coders - so you'll see something like the above, but much longer. You can filter to particular coder pairs using the
``coder1`` and ``coder2`` columns.

We can also compute the codebook's overall agreement across all coders and codes using ``pewanalytics``:

.. code:: python

    from pewanalytics.stats.irr import compute_overall_scores

    overall = compute_overall_scores(dataset, "document_id", "label_id", "coder_id")
    >>> overall

    {'alpha': 0.9114251886932071, 'fleiss_kappa': 0.9112031966849192}


Computing IRR for a single question
-----------------------------------

To calculate IRR for a single question, the process is exactly the same as above, except you just provide one
question in ``question_names``:

.. code:: python

    extractor = dataset_extractors["document_coder_label_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random"],
        "question_names": ["review_sentiment"],
    })
    dataset = extractor.extract(refresh=True)
    >>> dataset

    pk_x  coder_id coder_name  coder_is_mturk  document_id       date  label_id  ...                                               text  sampling_weight approx_weight strat_weight  keyword_weight  additional_weight document_type
       2         2     coder2           False           29 2000-01-30        11  ...  one-sided " doom and gloom " documentary about...              1.0           1.0         None            None               None  movie_review
       1         1     coder1           False           29 2000-01-30        11  ...  one-sided " doom and gloom " documentary about...              1.0           1.0         None            None               None  movie_review
       4         2     coder2           False           49 2000-02-19        11  ...   " the world on land -- it's just too big for ...              1.0           1.0         None            None               None  movie_review
       3         1     coder1           False           49 2000-02-19        11  ...   " the world on land -- it's just too big for ...              1.0           1.0         None            None               None  movie_review
       6         2     coder2           False           56 2000-02-26        11  ...  unfortunately it doesn't get much more formula...              1.0           1.0         None            None               None  movie_review

    scores = compute_scores_from_dataset(dataset, "document_id", "label_id", "coder_id", weight_column="sampling_weight")
    >>> scores

    coder1  coder2    n outcome_column  pos_label  coder1_mean  coder1_std  coder1_mean_unweighted  ...  accuracy        f1  precision    recall  precision_recall_min  matthews_corrcoef  roc_auc  pct_agree_unweighted
         2       1  200       label_id        NaN       10.945    0.016161                  10.945  ...      0.97  0.970000   0.970000  0.970000              0.970000             0.7114   0.8557                  0.97
         2       1  200   label_id__10        1.0        0.055    0.016161                   0.055  ...      0.97  0.727273   0.727273  0.727273              0.727273             0.7114   0.8557                  0.97
         2       1  200   label_id__11        1.0        0.945    0.016161                   0.945  ...      0.97  0.984127   0.984127  0.984127              0.984127             0.7114   0.8557                  0.97


Notice above that the ``label_id`` column contains values like 10 and 11 - these correspond to the primary keys of the
label values in the database. In the scores dataframe, the ``outcome_column`` shows you the overall IRR scores and then
code-level scores for each possible label in the question, as binary variables.

Computing binary IRR for a particular label
--------------------------------------------

An alternative way of getting the IRR scores for a particular label on a particular question - the positive class on a
binary question, for example, or perhaps a particular category from a list of topics - is to use a
``document_coder_dataset`` and then specify a particular column corresponding to a particular label value:

.. code:: python

    extractor = dataset_extractors["document_coder_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random"],
        "question_names": ["review_sentiment"],
    })
    >>> extractor.extract(refresh=True)

    document_id  coder_id  label_10  label_11 coder_name  coder_is_mturk  sampling_weight       date                                               text document_type
              0         1         0         1     coder1           False              1.0 2000-01-01  plot : two teen couples go to a church party ,...  movie_review
              0         2         0         1     coder2           False              1.0 2000-01-01  plot : two teen couples go to a church party ,...  movie_review
              4         1         0         1     coder1           False              1.0 2000-01-05  synopsis : a mentally unstable man undergoing ...  movie_review
              4         2         0         1     coder2           False              1.0 2000-01-05  synopsis : a mentally unstable man undergoing ...  movie_review
              5         1         0         1     coder1           False              1.0 2000-01-06  capsule : in 2176 on the planet mars police ta...  movie_review

    # Here we'll specifically look at the "label_10" column, and indicate that the positive label is 1:
    scores = compute_scores_from_dataset(dataset, 'document_id', 'label_10', 'coder_id', weight_column='sampling_weight', pos_label=1)
    >>> scores

    coder1  coder2    n outcome_column  pos_label  coder1_mean  coder1_std  coder1_mean_unweighted  ...  accuracy        f1  precision    recall  precision_recall_min  matthews_corrcoef  roc_auc  pct_agree_unweighted
         1       2  200    label_10__1          1        0.055    0.016161                   0.055  ...      0.97  0.727273   0.727273  0.727273              0.727273             0.7114   0.8557                  0.97

    # Since we only have one row of scores for the particular value we're interested in, and only two coders
    # we can also convert this into a dictionary:
    scores = scores.to_dict("records")[0]
    >>> scores

    {
        'coder1': 1,
        'coder2': 2,
        'n': 200,
        'outcome_column': 'label_10__1',
        'pos_label': 1,
        'coder1_mean': 0.055,
        'coder1_std': 0.016161092305986405,
        'coder1_mean_unweighted': 0.055,
        'coder1_std_unweighted': 0.016161092305986405,
        'coder2_mean': 0.055,
        'coder2_std': 0.016161092305986405,
        'coder2_mean_unweighted': 0.055,
        'coder2_std_unweighted': 0.016161092305986405,
        'alpha_unweighted': 0.7121212121212122,
        'cohens_kappa': 0.7113997113997114,
        'accuracy': 0.97, 'f1': 0.7272727272727273,
        'precision': 0.7272727272727273,
        'recall': 0.7272727272727273,
        'precision_recall_min': 0.7272727272727273,
        'matthews_corrcoef': 0.7113997113997114,
        'roc_auc': 0.8556998556998557,
        'pct_agree_unweighted': 0.97
    }


Computing IRR between two datasets
-----------------------------------

In some cases, you may want to calculate IRR between to separately-extracted dataset. For example, you could
extract a dataset of in-house coders that have been adjudicated, and another dataset of the Mechanical Turk
results for the same sample, collapsed by a particular threshold. In this example, we'll collapse Turkers with a
50% threshold, meaning that we'll require that at least half of the Turkers selected the same label for it to count,
otherwise we'll set the value to our negative class. Since we've adjudicated our in-house coders and therefore won't
have any ties because we're passing ``exclude_consensus_ignore=True``, we don't have to specify a threshold for
that dataset extractor like we do for the Mechanical Turk one.

.. code:: python

    from django_learning.models import Question
    base_class_id = (
        Question.objects\
            .filter(project__name="movie_reviews")
            .get(name="review_sentiment")
            .labels.get(value="0").pk
    )

    in_house = dataset_extractors["document_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random"],
        "question_names": ["review_sentiment"],
        "coder_aggregation_function": "mean",
        "coder_filters": [("exclude_mturk", [], {})],
        "convert_to_discrete": True,
        "exclude_consensus_ignore": True
    }).extract(refresh=True)

    mturk = dataset_extractors["document_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random"],
        "question_names": ["review_sentiment"],
        "coder_aggregation_function": "mean",
        "coder_filters": [("exclude_experts", [], {})],
        "convert_to_discrete": True,
        "threshold": 0.5,
        "base_class_id": base_class_id
    }).extract(refresh=True)


Now we'll use another Django Learning utility function, ``compute_scores_from_datasets_as_coders``, to
treat each dataset as a separate "coder" and calculate IRR between them:

.. code:: python

    from django_learning.utils.scoring import compute_scores_from_datasets_as_coders

    scores = compute_scores_from_datasets_as_coders(
        in_house,
        mturk,
        "document_id",
        "label_id",
        weight_column="sampling_weight",
        discrete_classes=True
    )
    >>> scores

      coder1    coder2    n outcome_column  pos_label  coder1_mean  coder1_std  coder1_mean_unweighted  ...  accuracy        f1  precision    recall  precision_recall_min  matthews_corrcoef   roc_auc  pct_agree_unweighted
    dataset1  dataset2  200       label_id        NaN        10.93    0.018087                   10.93  ...      0.97  0.966147   0.970938  0.970000              0.970000           0.744024  0.785714                  0.97
    dataset1  dataset2  200   label_id__10        1.0         0.07    0.018087                    0.07  ...      0.97  0.727273   1.000000  0.571429              0.571429           0.744024  0.785714                  0.97
    dataset1  dataset2  200   label_id__11        1.0         0.93    0.018087                    0.93  ...      0.97  0.984127   0.968750  1.000000              0.968750           0.744024  0.785714                  0.97

Now that we've got some coding data and the Mechanical Turkers look like they're doing a decent job, let's assume that
we pulled a larger sample and had them code that. Now let's see if we can
:doc:`train a classifier </tutorial/machine_learning>` using that data.


