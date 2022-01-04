from __future__ import print_function
import datetime
import pandas as pd

from django.test import TestCase as DjangoTestCase

from django_learning.models import *
from django_commander.commands import commands

from testapp.models import MovieReview
from testapp.utils import set_up_test_project


class TopicModelsTests(DjangoTestCase):

    """
    To test, navigate to django_learning root folder and run `python manage.py test testapp.tests`
    """

    def setUp(self):
        set_up_test_project()

    def test_topic_models(self):
        from django_learning.utils.topic_models import topic_models

        for val in ["test"]:
            self.assertIn(val, topic_models.keys())
            self.assertIsNotNone(topic_models[val]())

        commands["django_learning_topics_train_model"](name="test").run()
        model, _ = TopicModel.objects.get_or_create(name="test")
        model.load_model()  # refresh_model=False, refresh_vectorizer=False)
        t = model.topics.all()[0]
        t.anchors.extend(["action", "adventure", "thriller", "suspense", "surprise"])
        t.save()
        model.load_model(refresh_model=True)
        t = model.topics.exclude(anchors=[])[0]
        t.name = "action"
        t.label = "Mentions an action movie"
        t.save()
        commands["django_learning_topics_create_validation_coding_project"](
            topic_model_name="test",
            admin_name="coder1",
            project_hit_type="test_hit_type",
            create_project_files=True,
        ).run()
        # Stopping the testing here for now, since the modules need to be reloaded and can't be easily tested in one go
        # commands["django_learning_topics_create_validation_coding_project"](
        #     topic_model_name="test",
        #     admin_name="coder1",
        #     project_hit_type="test_hit_type",
        #     create_project_files=False,
        # ).run()

    def tearDown(self):

        from django.conf import settings
        import shutil

        for folders in [
            settings.DJANGO_LEARNING_PROJECTS,
            settings.DJANGO_LEARNING_SAMPLING_METHODS,
            settings.DJANGO_LEARNING_REGEX_FILTERS,
        ]:
            for f in folders:
                try:
                    os.unlink(os.path.join(f, "topic_model_test.json"))
                except:
                    pass
                try:
                    os.unlink(os.path.join(f, "topic_model_test.py"))
                except:
                    pass

        try:
            shutil.rmtree(os.path.join(settings.BASE_DIR, settings.LOCAL_CACHE_ROOT))
        except FileNotFoundError:
            pass
