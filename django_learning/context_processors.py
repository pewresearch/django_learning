from django.conf import settings


def identify_template(request):

    return {
        "django_learning_template": settings.DJANGO_LEARNING_BASE_TEMPLATE
    }