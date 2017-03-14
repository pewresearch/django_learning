Regex replacers
----------------

Regex replacers are used by the ``clean_text`` preprocessor to match and replace certain patterns of text. They must
contain a ``get_replacer`` function that returns a list of tuples, which themselves contain a pattern to match, and the
text to replace matches:

.. code:: python

    def get_replacer():

        return [
            (r"don\'t", "do_not")
        ]

Simply place this in a file in one of your ``settings.DJANGO_LEARNING_REGEX_REPLACERS`` folders.

You can pass a list of regex replacer names to the ``clean_text`` preprocessor like so:

.. code:: python

    from django_learning.utils.preprocessors import preprocessors

    text = "Regex replacers don't mess around."
    preprocessor = preprocessors["clean_text"](regex_replacers=["test"])
    >>> preprocessor.process(text)

    'regex replacers do_not mess'