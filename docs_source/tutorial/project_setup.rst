Project setup
==============

Defining a project
--------------------

To start a coding project, you first need to create a :doc:`Project codebook </utils/sampling_and_coding/projects>`.

Here's an example of a simple codebook. Let's say that we saved it in a file called ``movie_reviews.json``:

.. code:: json

    {
      "instructions": "Answer a question about a movie review",
      "qualification_tests": ["movie_test"],
      "admins": ["me"],
      "coders": ["me", "someone_else"],
      "questions": [
        {
          "prompt": "Is this review positive or negative?",
          "name": "review_sentiment",
          "display": "radio",
          "labels": [
            {"label": "Positive", "value": "1", "pointers": []},
            {"label": "Negative", "value": "0", "pointers": []}
          ],
          "tooltip": "",
          "examples": []
        }
      ]
    }

Defining a qualification test and scorer
-----------------------------------------

Since we also specified a qualification test in our projext JSON - which isn't necessary, but we're doing it here as an
example - we need to create a
:doc:`qualification test and scorer </utils/sampling_and_coding/project_qualification_tests_and_scorers>`.

Let's make it simple: we only want coders who watch movies. Let's create our qualification test and put it in a file
with the name that our project expects, ``movie_test.json``:

.. code:: json

    {
      "instructions": "See if you qualify to label movie reviews",
      "title": "Movie review qualification test",
      "description": "Answer the following questions to qualify",
      "price": 0.1,
      "approval_wait_hours": 24,
      "duration_minutes": 5,
      "lifetime_days": 7,
      "questions": [
        {
          "prompt": "Do you watch movies?",
          "name": "watch_movies",
          "display": "radio",
          "labels": [
            {"label": "Yes", "value": "1"},
            {"label": "No", "value": "0"}
          ]
        }
      ]
    }

If we're only doing in-house coding, some of the above parameters won't be used. In-house HITs are always free, for
example. But if we deploy HITs on Mechanical Turk, we'll be paying the Turkers 10 cents to complete the qualification
test.

Now we need to create a scorer for the test, in a file with the same name: ``movie_test.py``.

.. code:: python

    def scorer(qual_assignment):

        code = qual_assignment.codes.get(label__question__name="watch_movies")
        if int(code.label.value) == 1:
            return True
        else:
            return False

Creating a project in the database
----------------------------------

Now we're good to go!  As long as all of these files are in folders that are included in your Django Learning
settings, Django Learning will be able to find and use them.

To create your project in the database, you can run the following command:

.. code:: bash

    python manage.py run_command django_learning_coding_create_project movie_reviews

If we make changes to the project, we can just re-run this command, or access the project in the database and
call the ``save`` function:

.. code:: python

    from django_learning.models import Project
    Project.objects.get(name="movie_reviews").save()

Now that that's done, let's start :doc:`sampling </tutorial/sampling>`.