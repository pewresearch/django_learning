How Django Learning works
--------------------------

Installing
===========

To install Django Learning, simply add "django_learning" to your app's list of ``INSTALLED_APPS`` in ``settings.py``.
It should come before your own app, and after most everything else. You also need to install ``django_commander``.

.. code:: python

    INSTALLED_APPS = [
        "django.contrib.auth",
        "django.contrib.contenttypes",
        "django.contrib.staticfiles",
        "django_commander",
        "django_learning",
        "my_app",
    ]

Django Learning also provides an interface for coding and project management. To add these pages to your app, you
need to extend your ``urls.py`` file to add the Django Learning urls:

.. code:: python

    from django.conf.urls import include, url

    urlpatterns += [
        url(r"^learning/", include("django_learning.urls"))
    ]


Modular resources
==================

Django Learning is designed to be modular, allowing you to build a content analysis and machine learning
pipeline out of isolated and reusable components that follow a standard syntax with predictable behavior.
Everything, from project configuration files to machine learning feature extractors are stored in Python
or JSON files within your project, and they are automatically detected by Django Learning when it
initializes.  As you build your project files, you can organize them in a series of folders however
you like, and tell Django Learning where to look using variables specified in ``settings.py``. You can
store files in a nested structure of subfolders, and the subfolder names will be concatenated into a
unique name for each file as prefixes.  For example, if you wish to define a Django Learning Project -
a series of questions to be asked about a set of documents - you might develop your project in a JSON
file called `my_project.json` and put it in a folder in your Django app called ``learning/projects``.
To tell Django Learning where to look, you simply need to specify the location in ``settings.py``:

.. code:: python

    DJANGO_LEARNING_PROJECTS = [os.path.join(MY_APP_ROOT, "learning", "projects")]


Django Learning will then scan this - and any other folder specified by other compatible apps that you
have installed - extract the JSON for any projects it detects, and make them all available via
a single importable module:

.. code:: python

    from django_learning.utils.projects import projects
    projects['my_project']()


Django Learning has some base resources that are found in the ``django_learning.utils`` module,
which can be imported the same way:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors
    >>> dataset_extractors.keys()

    dict_keys([
        'raw_document_dataset',
        'document_coder_label_dataset',
        'model_prediction_dataset',
        'document_dataset',
        'document_coder_dataset'
    ])

As you develop custom modules for your content analysis pipeline, it's best to mirror the same
folder structure of the ``utils`` module in your own project. You can then update your ``settings.py``
with new variables pointing to the different types of plug-and-play modules you create:

.. code:: python

    DJANGO_LEARNING_DATASET_EXTRACTORS = [os.path.join(MY_APP_ROOT, "learning", "dataset_extractors")]

Assuming that the files you put in these subfolders conform to the required format, they'll become
accessible in Django Learning and will appear in the ``utils`` dictionaries. Let's say you created a
new dataset extractor in the folder above, in a file named ``my_dataset_extractor.py``:

.. code:: python

    from django_learning.utils.dataset_extractors import dataset_extractors
    >>> dataset_extractors.keys()

    dict_keys([
        'raw_document_dataset',
        'document_coder_label_dataset',
        'model_prediction_dataset',
        'document_dataset',
        'document_coder_dataset',
        'my_dataset_extractor'
    ])


Caching
========

Django Learning also caches the results of various plug-and-play resources, like dataset extractors and
stopword lists, as well as other things like machine learning models. Various forms of ``refresh=True``
can be used throughout Django Learning to recompute things, but Django Learning tries to use hashing
wherever possible to determine when it needs to refreshs things on its own. The caching can occur locally,
or you can configure Django Learning to use S3. For local caching, you need to add the following settings to
``settings.py``:

.. code:: python

    DJANGO_LEARNING_USE_S3 = False
    LOCAL_CACHE_ROOT = "cache"  # or wherever you want the cached data to be stored

For S3 caching, you need the following:

.. code:: python

    DJANGO_LEARNING_USE_S3 = True
    S3_BUCKET = "my_bucket_name"
    S3_CACHE_ROOT = "cache"  # or wherever you want the cached data to be stored

You also need to have environment variables set for "AWS_ACCESS_KEY_ID" and "AWS_SECRET_ACCESS_KEY".


Commands
========

A lot of things can be accomplished using Django Learning's built-in commands, which make use of
Django Commander. For example, once you've created a project (codebook) JSON file and put it in a
folder that Django Learning recognizes, you can create that project in the database by running
``python manage.py run_command django_learning_create_project MY_PROJECT_FILE_NAME``

Because of Django Commander, you can also access these commands programmatically and stitch together
your own project setup scripts, for example:

.. code:: python

    from django_commander.commands import commands
    commands["django_learning_create_project"](project_name="MY_PROJECT_FILE_NAME").run()

Now that we've got Django Learning installed, let's :doc:`create some documents </tutorial/documents>`.
