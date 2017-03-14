import pandas, copy

from django_pewtils import get_model

from django_learning.utils.feature_extractors import BasicExtractor


class Extractor(BasicExtractor):
    def __init__(self, *args, **kwargs):

        self.name = "topics"
        self.model = None
        self.topic_names = None

        super(Extractor, self).__init__(*args, **kwargs)

    def transform(self, X, **transform_params):

        X = copy.deepcopy(X)
        for p in self.get_preprocessors():
            X["text"] = X["text"].apply(p.run)

        topics = self.model.apply_model(X[["text"]]).fillna(0.0)
        del topics["text"]
        topics.columns = ["topic_{}".format(c) for c in topics.columns]
        topics = topics[self.topic_names]

        return topics

    def fit(self, X, y=None, **fit_params):

        self.model = get_model("TopicModel", app_name="django_learning").objects.get(
            name=self.params["model_name"]
        )
        self.topic_names = [
            "topic_{}".format(c)
            for c in list(
                self.model.topics.order_by("num").values_list("name", flat=True)
            )
            if c not in [None, ""]
        ]
        self.features = self.topic_names

        return self

    def get_feature_names(self):

        return self.topic_names
