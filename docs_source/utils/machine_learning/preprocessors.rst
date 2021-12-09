Preprocessors
==============

Preprocessors are Django Learning utilities that can be plugged into machine learning pipelines to clean up your
data before passing it to feature extractors. Django Learning feature extractors all take a ``preprocessors`` list.

Preprocessors are also just a handy way of storing utility code - you can access and use them programmatically
however you like, too!

Built-in preprocessors
----------------------

run_function
*************

This preprocessor runs an arbitrary ``function``. Super flexible.

.. code:: python

    text = "one two three"
    preprocessor = preprocessors["run_function"](function=lambda x: x[:3])
    >>> preprocessor.run(text)

    'one'


clean_text
***********

This preprocessor provides a wrapper around ``pewanalytics.text.TextCleaner``. In addition the kwargs accepted by that
class, it also accepts the following

    * ``stopword_sets``: a list of names corresponding to Django Learning stopword sets
    * ``stopword_whitelists``: a list of names corresponding to Django Learning stopword whitelists
    * ``regex_replacers``: a list of names corresponding to Django Learning regex replacers

.. code:: python

    from django_learning.utils.preprocessors import preprocessors

    text = "The cat jumped over the moon"
    preprocessor = preprocessors["clean_text"](
        process_method="lemmatize",
        stopword_sets=["english"]
    )
    >>> preprocessor.run(text)

    'cat jumped moon'


expand_text_cooccurrences
***************************

This function takes all of the words in the document and returns all possible two-word (bigram) combinations. This
can be useful for filling out the feature space in sparse documents, where the use of two words together in a document
might be meaningful even if they don't occur next to each other.

.. code:: python

    text = "one two three"
    preprocessor = preprocessors["expand_text_cooccurrences"]()
    >>> preprocessor.run(text)

    'one two one three two three'

r