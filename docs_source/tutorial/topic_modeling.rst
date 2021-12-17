Topic modeling
==================

To train a topic model, you need to define a :doc:`config file </utils/machine_learning/topic_models>` that specifies
the topic model's parameters, including the scope of the documents to train on (a sampling frame), the vectorizer to
use, etc. Let's specify a topic model and put it in a file called "movie_review_themes.py":

.. code:: python

    def get_parameters():

        return {
            "frame": "movie_reviews",
            "num_topics": 5,
            "sample_size": 1000,
            "anchor_strength": 4,
            "vectorizer": {
                "sublinear_tf": False,
                "max_df": 0.9,
                "min_df": 10,
                "max_features": 8000,
                "ngram_range": (1, 2),
                "use_idf": False,
                "norm": None,
                "binary": True,
                "preprocessors": [
                    (
                        "clean_text",
                        {
                            "process_method": ["lemmatize"],
                            "regex_filters": [],
                            "stopword_sets": ["english"],
                            "stopword_whitelists": [],
                        },
                    )
                ],
            },
        }

We'll train it on a sample of 200 documents from our movie review sampling frame. Now that we've specified the config
file, we can use the ``django_learning_topics_train_model`` command to actually train the model.

.. code:: bash

    python manage.py run_command django_learning_topics_train_model

Now, if we hop into a Django shell, we can grab the model from the database and take a look:

.. code:: python

    from django_learning.models import TopicModel

    model, _ = TopicModel.objects.get_or_create(name="test")
    >>> model.topics.all()

    <BasicExtendedManager [
        <Topic: starring directed cinema robert adaptation novel tim based self peter strong company american story short>,
        <Topic: york new_york high_school office box_office years_ago box school ago town high new years small coming called happy world unfortunately>,
        <Topic: susan susan_granger granger granger_review science_fiction fiction motion_picture motion science note_consider spoilers_forewarned portions_following portions text_spoilers forewarned following_text consider_portions text picture note>,
        <Topic: films plot little bad horror and early lot right>,
        <Topic: screen director writer saw series remake smith bob night william rating cast classic seen>
    ]>


    >>> for t in model.topics.all(): print("{}: {}".format(t.num, t))

    4: starring directed cinema robert adaptation novel tim based self peter strong company american story short
    3: york new_york high_school office box_office years_ago box school ago town high new years small coming called happy world unfortunately
    2: susan susan_granger granger granger_review science_fiction fiction motion_picture motion science note_consider spoilers_forewarned portions_following portions text_spoilers forewarned following_text consider_portions text picture note
    1: films plot little bad horror and early lot right
    0: screen director writer saw series remake smith bob night william rating cast classic seen


Looks like some of these make sense, but they're not great. Since Django Learning uses
CorEx, which is a semi-supervised model, we can actually nudge our model a bit, and see if we can make some
improvements. If you open up the Django Learning interface, you'll actually be able to access this model in the
"Topic Models" section, and add anchor terms to the different topics in the model to try to nudge them towards terms
that are more meaningful or interesting to you. The interface also allows you to give a label or name to each topic,
giving it an interpretation that we can later validate. Let's see if we can suggest to the model some more
interesting topics...

.. code:: python

    >>> for t in model.topics.all(): print("{}: {}".format(t.num, t.anchors))

    4: []
    3: ['horror', 'scary', 'suspense']
    2: ['scifi', 'sci fi', 'science fiction', 'science']
    1: ['comedy', 'funny', 'humor']
    0: ['action', 'adventure']

    model.load_model(refresh_model=True)

    >>> for t in model.topics.all(): print("{}: {}".format(t.num, t))

    4: plot make good is_not movies
    3: horror susan susan_granger granger_review granger high_school school review high genre look surprise classic peter coming million black unfortunately house
    2: science science_fiction fiction motion_picture motion spoilers_forewarned portions_following portions note_consider text_spoilers consider_portions forewarned following_text text note picture consider spoilers following making
    1: comedy funny romantic characters quite entertaining thought family minutes expect
    0: action live disney effects john special van way kind god animated

Now we've got some interesting starting points for topics about different genres: action, comedy, etc. But there's
still a lot of noise here. Let's see if we can clean those topics up more. To do this, we're actually going to
sacrifice topic 4 - by filling its anchor list full of words from the other topics that don't really seem to belong,
or that we don't care about. Topid 4 is going to become our "junk topic" - somewhere where we can tell the model to
put words that we don't want appearing in the other topics. Let's retrain our model and see
what effect this has:

.. code:: python

    >>> for t in model.topics.all(): print("{}: {}".format(t.num, t.anchors))

    4: ['starring', 'directed', 'director', 'novel', 'writer', 'friends', 'synopsis', 'disney', 'motion', 'picture', 'quite', 'question', 'text', 'following', 'spoilers', 'consider', 'note', 'susan', 'granger', 'review', 'box', 'office', 'school', 'high', 'new', 'york', 'motion', 'portions', 'forewarned', 'text', 'note', 'following', 'susan granger', 'peter', 'look', 'granger review']
    3: ['horror', 'scary', 'suspense']
    2: ['scifi', 'sci fi', 'science fiction', 'science']
    1: ['comedy', 'funny', 'humor']
    0: ['action', 'adventure']

    >>> for t in model.topics.all(): print("{}: {}".format(t.num, t))

    4: review picture spoilers school following susan granger granger_review susan_granger consider novel note box text office motion forewarned portions high_school motion_picture
    3: horror films movies film dumb house scream budget surprise comes
    2: science science_fiction fiction special genre making popular career martin tell recent
    1: comedy funny romantic characters entertaining and tim thought feel family minutes expect
    0: action live effects john van way kind thriller god mystery

Alright, these are starting to look a little better. Our "horror" topic is starting to pull in words like "scream" and
"surprise", our "action" topic now has the word "thriller" - this example isn't great because we aren't working with
a whole lot of data, but you can get the idea. By iteratively adding anchor terms and retraining the model, we can
improve our topics and eventually arrive at something that seems interpretable.

Once our topics are in a good place, we can give the ones we like specific labels. For example, topic 0 might be
equivalent to a movie review that "Mentions an action movie". Let's set this as a label:

.. code:: python

    topic = model.topics.get(num=0)
    topic.name = "action"
    topic.label = "Mentions an action movie"
    topic.save()
    model.load_model(refresh_model=True)

Once we've given labels to all of the topics that look interesting, we can use the
``django_learning_topics_create_validation_coding_project`` command to create a coding project and pull a sample to
validate the topic model. You can first run this command with ``create_project_files=True`` to create and save a
codebook file automatically. The codebook will consist of a question for each labeled topic in the model, asking
whether or not the document mentions that  topic or not. The command will also make a custom sampling method that
pulls a sample where 50% of the documents match to at least one of the anchor terms in the model.

Once you've created the project file, you can then run the command without the ``--create_project_files`` option,
and specify a ``sample_size`` and ``num_coders`` and coding samples and HITs will be created automatically for you,
oversampled on the topic model's anchor terms.

Once you've coded your sample, you can then run the ``django_learning_topics_assess_irr`` command to calculate
how well the model actually measures the topics you think it does, based on the interpretations you gave it. If it
looks good, you're good to apply the model and use it to code documents.