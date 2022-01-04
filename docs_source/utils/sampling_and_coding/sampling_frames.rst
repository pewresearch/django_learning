Sampling frames
===================

Sampling frames specify a scope around a set of documents that you wish to sample from. Sampling frames
should represent a _population_ of documents from which you will draw representative samples. The scope of
sampling frames are defined in a configuration file that should be included in one of your
``settings.DJANGO_LEARNING_SAMPLING_FRAMES`` folders. The file should include a ``get_frame()`` function
that returns a dictionary like so, with the following properties:

.. code:: python

    def get_frame():

        return {
            "filter_dict": {"movie_review__isnull": False},
            "exclude_dict": {},
            "complex_filters": [],
            "complex_excludes": []
        }

Parameters
-----------

filter_dict
************

This parameter contains a dictionary that filters documents using Django query parameters. Usually this will
contain filters on relations between Documents and models in your own app, as in the above example. The dictionary
will be passed to a standard Django query set filter like so:

.. code:: python

    filter_dict = {"some_field": "some_value"}
    Document.objects.filter(**filter_dict)

exclude_dict
*************

Same as filter_dict, except it gets passed to ``Document.objects.exclude``.

complex_filters
****************

A list of Django query set filters that are passed directly to ``.filter`` like so:

.. code:: python

    complex_filters = [
        Q(some_field=some_value) | Q(some_other_field=some_other_value)
    ]
    docs = Document.objects.all()
    for c in complex_filters:
        docs = docs.filter(c)

This can be useful for passing complex queries using ``django.db.models.Q``, etc.

complex_excludes
*****************

Same as complex_filters, but they get passed to ``Document.objects.exclude``.

Creating sampling frames
-------------------------

If you've created a sampling frame config file, you then need to create it in the database. To do this, you can
use the ``django_learning_coding_extract_sampling_frame`` command, or you can create it direclty in the database.
To actually run the queries and associate documents with the frame, you need to run the ``extract_documents`` function
(which gets run automatically with the extraction command mentioned above).

.. code:: python

    from django_learning.models import SamplingFrame

    frame, _ SamplingFrame.objects.get_or_create(name="my_frame")
    frame.extract_documents()

Once a frame has been extracted, the documents associated with it become fixed until you run
``frame.extract_documents(refresh=True)``. It is best to extract frames only once the set of documents you wish to
study has been finalized. If you refresh a frame with existing samples and the scope of the frame changes to add or
drop documents, the samples may no longer be representative of the updated frame. If your samples have stratification
variables that capture the reasons behind changes in a frame's scope, the
``django_learning.utils.sampling.update_frame_and_expand_samples`` function can be used to expand the frame and
existing samples accordingly, but it's highly recommended to consider frames "locked down" and finalized before you
start pulling samples.

Built-in frames
----------------

``all_documents``
*******************

This built-in sampling frame simply matches to all existing Document objects in the database.
