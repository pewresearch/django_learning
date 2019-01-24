from django.conf import settings

from django_pewtils import get_model


def identify_template(request):

    return {
        "django_learning_template": settings.DJANGO_LEARNING_BASE_TEMPLATE
    }

def get_document_classification_model_names(request):

    return {
        "document_classification_models": get_model("DocumentClassificationModel", app_name="django_learning").objects.values_list("name", flat=True)
    }