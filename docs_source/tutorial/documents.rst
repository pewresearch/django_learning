Documents
==========

Django Learning is all about coding things. The main objects that get coded are :doc:`documents </models/documents>`.
Documents have text, although you can adapt Django Learning to display images, videos, audio or whatever else you
want to code by using :doc:`custom HIT templates </utils/sampling_and_coding/project_hit_templates>`.

Before you can do anything with Django Learning, then, you need to populate the database with documents that can
be sampled. To do this, you need to create a relation between ``django_learning.models.Document`` and a model in your
own app. For example, let's say we have a database of movie reviews, stored in a ``MovieReview`` model in our app.
To hook Django Learning up, we need to associate this model with documents. Let's give each movie review a one-to-one
relationship with a document:

.. code:: python

    from django.db import models

    class MovieReview(models.Model):

        publication = models.ForeignKey("myapp.Publication", related_name="movie_reviews")
        review_text = models.TextField()
        document = models.OneToOneField(
            "django_learning.Document",
            related_name="movie_review",
            on_delete=models.SET_NULL,
            null=True,
        )

In some cases, you may already have text stored on your own model in other fields. It's up to you to decide whether
you want to move that data to the documents, or simply duplicate it. To do the former, we'd just delete the
``review_text`` field and make a migration after we've populated all our documents, and then modify the code that we
use to populate the ``MovieReview`` table to automatically create new documents in the future. For this example, we'll
just copy it over:

.. code:: python

    from my_app.models import MovieReview
    from django_learning.models import Document

    for review in MovieReview.objects.all():
        doc, _ = Document.objects.get_or_create(movie_review=review)
        doc.text = review.review_text
        doc.save()

Once we've done this, it's a good idea to update our app to create documents whenever movie reviews are created,
either in our data collection script, or by extending the ``save()`` function to do this automatically:

.. code:: python

    from django.db import models
    from django_learning.models import Document

    class MovieReview(models.Model):

        review_text = models.TextField()
        document = models.OneToOneField(
            "django_learning.Document",
            related_name="movie_review",
            on_delete=models.SET_NULL,
            null=True,
        )

        def save(self, *args, **kwargs):
            super(MovieReview, self).save(*args, **kwargs)
            self.document, _ = Document.objects.get_or_create(movie_review=self)
            self.document.text = self.review_text
            self.document.save()

You can create relations between the ``Document`` model and as many models in your app as you'd like. You can then
filter the document table on these relations to select different types of documents:

.. code:: python

    Document.objects.filter(movie_review__isnull=False)

Anyway, we've now got a bunch of documents in our database. Let's go
:doc:`create a coding project </tutorial/project_setup>`.