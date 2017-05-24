from django.conf import settings


def identify_template(request):

    print settings.DJANGO_LEARNING_BASE_TEMPLATE
    return {
        "django_learning_template": settings.DJANGO_LEARNING_BASE_TEMPLATE
    }