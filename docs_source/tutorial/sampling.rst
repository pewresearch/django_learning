Sampling
=========

Creating a sampling frame
-------------------------

Now that we have documents created in the database, and project with questions we want to code, we need to define
a :doc:`sampling frame </utils/sampling_and_coding/sampling_frames>`. To do this, let's create a ``movie_reviews.py``
file that provides specifications for selecting our movie review universe of documents:

.. code:: python

    def get_frame():
        return {
            'filter_dict': {'movie_review__isnull': False}
        }

Having done that, all we need to do now is extract the sampling frame in our database. We can run the following
command to do so:

.. code:: bash

    python manage.py run_command django_learning_coding_extract_sampling_frame movie_reviews

That command will create a :doc:`SamplingFrame </models/sampling>` object in the database and associate it with all of
the documents that match to the filter we wrote.

Creating coders
----------------

Django Learning uses the :doc:`Coder </models/coders>` model to track the people who code your projects. It's a simple
model where each coder is given a name, and is linked to ``django.contrib.auth.models.User``. You can create coders
using the ``django_learning_coding_create_coder`` command, which will create a coder with the specified name in the database, and
will give them the default password of "pass". Since Django Learning is meant to be deployed internally, the purpose
of these accounts is simply for tracking coding behavior - security isn't a concern. Let's create the two coders
that will be doing work on our project:

.. code:: bash

    python manage.py run_command django_learning_coding_create_coder me
    python manage.py run_command django_learning_coding_create_coder someone_else

Defining a sampling method
--------------------------

If you want to do more complex sampling that goes above and beyond a random sample, you'll need to define a
:doc:`sampling method </utils/sampling_and_coding/sampling_methods>`. Let's say that we have movie reviews from
a few different publications in our database, which are related to our ``MovieReview`` model via a ``publication``
foreign key. To ensure that we sample smoothly across the different publications, we can specify a custom
sampling method to stratify across them:

.. code:: python

    def get_method():

        return {
            "sampling_strategy": "random",
            "stratify_by": "movie_review__publication",
            "sampling_searches": [],
            "additional_weights": {}
        }

Let's put this in a file called ``stratify_by_publication.py``.

Defining a regex filter for oversampling
*****************************************

If we want to take it a step further, we could also set our sampling method up for keyword oversampling.
Let's say that we're particularly interested in coding action movies, but they're relatively rare in our
sampling frame, so we want to oversample reviews with terms related to action movies to boost their prevalence
in our sample. To do this, we'd first need to create a :doc:`regex filter </utils/multipurpose/regex_filters>` to
identify documents that contain those terms. Let's make a regex filter and put it in a file called ``action.py``:

.. code:: python

    import re

    def get_regex():
        return re.compile(r"action|adventure", flags=re.IGNORECASE)

Now let's return to our sampling method. Let's add this regex filter and specify that we want at 20% of the documents
in our sample to match to the regex:

.. code:: python

    def get_method():

        return {
            "sampling_strategy": "random",
            "stratify_by": "movie_review__publication",
            "sampling_searches": [{"regex_filter": "action", "proportion": 0.2}],
            "additional_weights": {}
        }

Since this goes above and beyond stratifying by publication, let's put this in a different sampling method file,
``stratify_by_publication_and_oversample_action.py``.

Extracting a sample
--------------------

Now that we have some custom sampling methods defined, we can pull samples using the
``django_learning_coding_extract_sample`` command. To do a simple random sample, we can use Django Learning's
built-in "random" sampling method. We'll call this sample ``movie_review_sample_random`` and pull 100 documents:

.. code:: bash

    python manage.py run_command django_learning_coding_extract_sample movie_reviews movie_review_sample_random --sampling_frame_name movie_reviews --sampling_method random --size 100

Let's also pull another sample of the same size, but this time we'll use the
``stratify_by_publication_and_oversample_action`` method. We'll call this sample ``movie_review_sample_oversample``:

.. code:: bash

    python manage.py run_command django_learning_coding_extract_sample movie_reviews movie_review_sample_oversample --sampling_frame_name movie_reviews --sampling_method stratify_by_publication_and_oversample_action --size 100

Now we have two samples extracted in our database. Now it's time to :doc:`create some HITs </tutorial/coding>`.

Recommended sampling strategy
------------------------------

Usually when you're doing a content analysis project, you want to develop a codebook iteratively until you
achieve IRR across multiple coders, and then you want to divvy up the remaining documents and have the coders
divide-and-conquer and code the rest of them individually. Alternatively, you may have the coders code a larger
sample, and then use that sample as training data to train a classifier, which you will then use to code the
remaining documents in your sampling frame.

Either way, it's recommended to iteratively pull small samples and make adjustments to your codebook until you
get good IRR. When you do, keep that final sample and adjudicate disagreements - you now have a baseline.
If you wish to do coding on Mechanical Turk, this is the perfect point of comparison to use. Create HITs for your
final IRR sample on Mechanical Turk, and then test out different thresholds to determine what maximizes
IRR with your in-house gold standard. (See :doc:`Computing IRR </tutorial/computing_irr>` for more info.)

If the Turkers look good, you can then pull another sample - either a larger one, or the full remainder of the
sampling frame using the ``all_documents`` sampling method. You can then either create in-house HITs with
``num_coders=1`` to have your coders divvy up the remaining documents, or you can create Mechanical Turk HITs
the same as you did for the IRR sample, and have them code the remainder for you, no further in-house coding
required. This larger sample can then be plugged in to a machine learning pipeine using a ``document_dataset``
extractor, and you can even use the original IRR sample as the ``test_dataset_extractor`` to compare the model
directly against your established gold standard (see :doc:`Pipelines </utils/machine_learning/pipelines>` for more.)
