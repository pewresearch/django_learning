Extracting datasets and collapsing coders
==========================================

Once you have coding data, you can extract it using one of the built-in Django Learning
:doc:`dataset extractors </utils/dataset_extraction/dataset_extractors>`.

To get all of the coding data for your project, across multiple question and coders, you can use a
``document_coder_label_dataset`` extractor:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    extractor = dataset_extractors["document_coder_label_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random"],
        "question_names": ["watch_movies", "another_question"]
    })
    >>> extractor.extract(refresh=True)

To get a coder-level dataset for a specific question - which is useful for calculating IRR - you can use a
``document_coder_dataset``.

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors

    extractor = dataset_extractors["document_coder_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random"],
        "question_names": ["watch_movies"]
    })
    >>> extractor.extract(refresh=True)

Normally, the point of coding is to arrive at a dataset for analysis, with one row per document. To extract such a
dataset, you can use a ``document_dataset`` extractor. For samples where only one coder coded each document, you'll
naturally get such a dataset. For samples with more than one coder per dataset, you can either A) adjudicate
disagreements in the interface and then pass ``exclude_consensus_ignore=True`` to the dataset extractor, for in-house
HITs, or B) you can specify a threshold for collapsing multiple coders into a single value for each document.
This is detailed more in the :doc:`dataset extractors section</utils/dataset_extraction/dataset_extractors>`.

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors
    from django_learning.models import Question
    base_class_id = (
        Question.objects\
            .filter(project__name="movie_reviews")
            .get(name="watch_movies")
            .labels.get(value="0").pk
    )

    extractor = dataset_extractors["document_dataset"](**{
        "project_name": "movie_reviews",
        "sample_names": ["movie_review_sample_random"],
        "question_names": ["watch_movies"],
        "coder_aggregation_function": "mean",
        "convert_to_discrete": True,
        "threshold": 0.5,
        "base_class_id": base_class_id
    })
    >>> extractor.extract(refresh=True)
