from tqdm import tqdm

from django.db.models import Count


def get_stopwords():

    from django_learning.models import Entity

    stopwords = []
    for e in tqdm(
        Entity.objects.annotate(c=Count("documents")).filter(c__gte=10),
        desc="Adding entity stopwords",
    ):
        if len(e) > 2:
            stopwords.append(e)
        # stopwords.extend(e.name.lower().split(" "))
    return list(set(stopwords))
