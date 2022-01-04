Stopword sets
=============

Stopword sets define a set of words to exclude from your document text. They're most useful when included as
part of a machine learning pipeline by passing them to the ``clean_text`` preprocessor. They should be defined in
files placed in one of your ``settings.DJANGO_LEARNING_STOPWORD_SETS`` folders and each file should contain a
``get_stopwords`` function that returns a list of strings. Because these are contained in functions, you can calculate
lists of stopwords dynamically based off of values in your database.

.. code:: python

    from my_app.models import Person

    def get_stopwords():
        return list(Person.objects.values_list("last_name", flat=True))

Caching
---------

Because they can be based off of time-consuming queries, stopword lists get automatically cached by the ``clean_text``
preprocessor. The assumption is that they don't change very often. You can force the preprocessor to refresh the
stopword lists by passing it ``refresh_stopwords=True``, either directly:

.. code:: python

    from django_learning.utils.preprocessors import preprocessors
    preprocessors["clean_text"](stopword_sets=["my_stopword_list"], refresh_stopwords=True)

Or, better yet, you can put it in your machine learning pipeline by placing ``"refresh_stopwords": True`` alongside
your "stopword_sets" parameter. Once you've got your pipeline locked down, you can then switch this to False and
the stopwords will remain fixed, which is good behavior anyway, because once you've trained your model and are ready
to apply it, you don't want the feature space to change anymore.
