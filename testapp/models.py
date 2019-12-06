from django.db import models

from django_pewtils.abstract_models import BasicExtendedModel


class MovieReview(BasicExtendedModel):

    document = models.OneToOneField(
        "django_learning.Document",
        related_name="movie_review",
        on_delete=models.SET_NULL,
        null=True,
    )
