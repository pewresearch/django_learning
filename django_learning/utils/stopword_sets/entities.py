from tqdm import tqdm

from django.db.models import Count


def get_stopwords():

    from django_learning.models import Entity

    stopwords = []
    for e in tqdm(
        Entity.objects.annotate(c=Count("documents")), desc="Adding entity stopwords"
    ):
        if len(e.name) > 2:
            stopwords.append(e.name.lower())

    return list(set(stopwords))
