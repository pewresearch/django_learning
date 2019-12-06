from __future__ import absolute_import

import random

from django.db.models import F, Q

from django_pewtils import get_model


def get_frame():

    return {
        "filter_dict": {"text__regex": "action"},
        "exclude_dict": {"text__regex": "adventure"},
        "complex_filters": [],
        "code_weights": [],
    }
