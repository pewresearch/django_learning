Stopword whitelists
====================

Stopword whitelists operate similarly to stopword sets, except they indicate words that you do NOT want to exclude
from the feature space. This is most useful when you're using a stopword set that you're compiling from your database,
and there are a handful of terms in the stopword set that you want to exclude. You could modify your stopword set to do
this directly, but if you're using multiple dynamic stopword sets, it's probably safer to hard-code a list of words that
you've decided you always want to include. Stopword whitelists should be defined in files in one of your
``settings.DJANGO_LEARNING_STOPWORD_WHITELISTS`` folders and the files should contain ``get_whitelist`` functions:

.. code:: python

    def get_whitelist():
        return [
            "cats"
        ]

As with stopword sets, these are most commonly placed in the "stopword_whitelists" argument that gets passed to
the ``clean_text`` preprocessor. However, as will all plug-and-play Django Learning utils, you can always access
it directly like so:

.. code:: python

    from django_learning.utils.stopword_whitelists import stopword_whitelists

    >>> stopword_whitelists["my_whitelist"]()
    ["cats"]