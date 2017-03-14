Coding
========

Defining a custom HIT template
-------------------------------

By default, Django Learning will use the default HIT template that can be found in
``django_learning/templates/django_learning/hit.html``. This template displays the document's text alongside all of the
questions in your project, and contains a bunch of logic for displaying different question types, handling
the visibility of dependencies, adding tooltips and popup modals, and hooking things into Mechanical Turk. If you
want to make modifications to how your coding project gets displayed, you can copy this template into your own
HTML file, located in one of your ``settings.DJANGO_LEARNING_PROJECT_HIT_TEMPLATES`` folders, and make whatever changes
to the layout you'd like. Things we've done in the past include adding images and videos, color-coded annotations to
the document text, embedding tweets, and adding external links to things like news articles and Facebook posts.

Creating a HIT type
--------------------

Next, we need to create a JSON defining a :doc:`HIT type </utils/sampling_and_coding/project_hit_types>`, which is
mostly configuration for how Mechanical Turk will list your tasks and who can do them. Let's create a simple HIT type
in a file called ``movie_review.json``:

.. code:: json

    {
      "title": "Read some movie reviews",
      "description": "Read movie reviews and answer some questions",
      "keywords": ["labeling", "movies"],
      "price": 0.1,
      "approval_wait_hours": 24,
      "duration_minutes": 10,
      "lifetime_days": 7,
      "min_approve_pct": 0.95,
      "min_approve_cnt": 100
    }


Creating in-house HITs
------------------------

Now that we have our HIT type defined, and our samples extracted, we can create coding assignments! There's a built-in
command that makes this easy. Let's create HITs for our ``movie_review_sample_random`` sample and specify that we
want two coders to code each movie review. We'll pass in the name of our project, sample and HIT type:

.. code:: bash

    python manage.py run_command django_learning_coding_create_sample_hits movie_reviews movie_review_sample_random movie_review --num_coders 2

If we want to use our custom HIT template, we can pass in the path to our HTML file via ``--template_name``.

Once you run this command, coders should be able to log into the interface and start coding! Make sure you've added
the Django Learning urls to your ``urlpatterns`` in ``urls.py``, and then you should be able to navigate to your sample
using the url ``/learning/project/<project name>/sample/<sample name>``.


Creating Mechanical Turk HITs
------------------------------

If you want to have Mechanical Turkers code your sample in addition to (or instead of) in-house coders, it's easy to
do so. The process is much the same as above, you just use a different command:

.. code:: bash

    python manage.py run_command django_learning_coding_mturk_create_sample_hits movie_reviews movie_review_sample_random movie_review --num_coders 2

Syncing Mechanical Turk HITs
-----------------------------

As Turkers complete their assignments, you'll need to hit the API and download their results. To do this, you can
use the following command to sync with the API:

.. code:: bash

    python manage.py run_command django_learning_coding_mturk_sync_sample_hits movie_reviews movie_review_sample_random

By default, this command will simply loop over your HITs and pull data for any newly-completed assignments. It can
be useful to run this command on an infinite loop while Turkers are completing their assignments, so you can track
progress in the interface. To do this, you can just pass the ``--loop`` option.

Eventually you'll need to pay your Turkers for their work. Doing so automatically as they complete their assignments
isn't recommended, though, since it gives them the impression that they're getting approved automatically and aren't
being checked for the quality of their work. Instead, it's useful to set this command up on ``--loop`` and specify the
``--approve`` flag with an ``--approve_probability`` option that will randomly approve assignments with a certain
probability on each loop. Doing this during daylight hours gives the impression that we're reviewing the assignments,
and keeps Turkers honest. Once all of the assignments are done, you can then remove the ``approve_probability``
option and approve all of the unpaid assignments. Let's set up a loop where we'll hit the API every 60 seconds and
approve assignments with a 10% probability:

.. code:: bash

    python manage.py run_command django_learning_coding_mturk_sync_sample_hits movie_reviews movie_review_sample_random --loop --approve --approve_probability .1 --time_sleep 60

Switching out of sandbox mode
------------------------------

By default, when you create a project, it has a flag called ``mturk_sandbox`` that's set to ``True``. When this flag
is enabled, you'll be using the Mechanical Turk sandbox API. If you tried doing the above, you'd have noticed that
no assignments were getting completed even after you'd been syncing a while - that's because you were using the sandbox
and you would have to use the sandbox to view and code the HITs yourself for anything to show up. Once you've finished
testing things out and you're ready to deploy the HITs to Mechanical Turk for real, you can switch your project out of
sandbox mode like so:

.. code:: bash

    python manage.py run_command django_learning_coding_mturk_exit_sandbox movie_reviews

This is a permanent change; once you've created live HITs on Mechanical Turk for any of the samples attached to your
project, you can't switch the project back to sandbox mode. You'd have to create a new project and go through the
process again.

Adjudicating disagreements
---------------------------

For in-house coding, if you have multiple coders completing each HIT, it can be useful to go through their
disagreements to better understand how they're coding, and to reconcile those disagreements to arrive at a single
correct code for each document. When viewing your sample in the Django Learning interface,
admins have the ability to access an "Adjudicate disagreements" section.

Clicking on the name of a particular question will bring you to a queue of side-by-side disagreements on that
question, allowing you to review disagreements in random order and select which of the two are correct. Later, when using
one of the built-in Django Learning :doc:`dataset extractors </utils/dataset_extraction/dataset_extractors>`, you can
then pass ``exclude_consensus_ignore=True`` to remove the codes that were incorrect (the ones you did not choose as
correct) from the dataset. This can be particularly useful if you pull a sample for computing IRR - after adjudicating
disagreements and arriving at the correct "gold standard" codes, the sample can then be used as a test dataset on a
machine learning pipeline to evaluate the model against that gold standard.

Reviewing "uncodeable" HITs
----------------------------

Admins also have the option of reviewing and making corrections to assignments that were marked "uncodeable" by
in-house coders. On the sample screen, admins should see buttons next to each coder labeled "Review Uncodeable";
clicking on these will bring you to a screen that allows you to make corrections. As you do so, if you uncheck the
"uncodeable" checkbox before saving, you can clear out the queue and the buttons will eventually disappear.

Making corrections manually
----------------------------

In the sample screen, admins also have the ability to "View Assignments" for any coder on the project. Clicking on this
brings you to a screen with all of their completed assignments, and allows the admin to click on a particular
assignment and make corrections as the original coder.
