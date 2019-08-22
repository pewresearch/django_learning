from __future__ import print_function
from builtins import str
import pandas, copy, itertools

from pewtils import is_null, is_not_null
from django_pewtils import get_model

from django_learning.utils.scoring import compute_scores_from_dataset
from django_commander.commands import BasicCommand, log_command, commands


def get_topic_base_class(topic):

    try:
        base_class_id = get_model("Question", app_name="django_learning").objects \
            .filter(project__name="topic_model_{}".format(topic.model.name)) \
            .get(name=topic.name) \
            .labels.get(value="0").pk
    except:
        base_class_id = None

    return base_class_id


def get_expert_topic_code_extractor_params(topic, coder_filters, exclude_consensus_ignore=False):

    return {
        "project_name": "topic_model_{}".format(topic.model.name),
        "sample_names": ["topic_model_{}".format(topic.model.name)],
        "question_names": [topic.name],
        "document_filters": [],
        "coder_filters": coder_filters,
        "base_class_id": get_topic_base_class(topic),
        "threshold": .4,
        "convert_to_discrete": True,
        "balancing_variables": [],
        "ignore_stratification_weights": False,
        "exclude_consensus_ignore": exclude_consensus_ignore
    }


def get_topic_expert_codes(dataset_name, topic, sample_name, coder_filters, exclude_consensus_ignore=False, refresh_codes=False):

    from django_learning.utils.dataset_extractors import dataset_extractors
    try:
        codes = dataset_extractors["document_dataset"](
            **get_expert_topic_code_extractor_params(topic, coder_filters, exclude_consensus_ignore=exclude_consensus_ignore)
        ).extract(refresh=refresh_codes)
    except Exception as e:
        print(e)
        codes = None

    if is_not_null(codes) and len(codes) > 0:

        base_class_id = get_topic_base_class(topic)
        codes['value'] = codes['label_id'].map(lambda x: 1 if str(x) != str(base_class_id) else 0)
        codes["dataset"] = dataset_name
        codes["probability"] = None
        codes["sampling_weight"] = codes["sampling_weight"].astype(float)
        codes["value"] = codes["value"].astype(str)
        codes = codes[["document_id", "value", "dataset", "sampling_weight", "probability", "text"]]

    return codes


def get_topic_dictionary_codes(topic, dataset, anchor_override=None):

    from django_learning.utils.feature_extractors import feature_extractors

    if is_null(dataset):
        return None

    dictionary_codes = copy.copy(dataset)
    dictionary_codes['dataset'] = 'dictionary'

    ngram_set = get_model("NgramSet", app_name="django_learning").objects.create_or_update(
        {"dictionary": "topic_model_{}".format(topic.model.name), "name": topic.name},
        {"label": topic.label[:99], "words": topic.anchors if not anchor_override else anchor_override}
    )
    extractor = feature_extractors["ngram_set"](
        dictionary="topic_model_{}".format(topic.model.name),
        include_ngrams=False,
        ngramset_name=topic.name,
        preprocessors=topic.model.parameters["vectorizer"]["preprocessors"]
    )
    extractor.fit(dictionary_codes)
    dictionary_codes[topic.name] = extractor.transform(dictionary_codes)
    dictionary_codes["value"] = dictionary_codes[topic.name].map(lambda x: "1" if x > 0 else "0")
    if "probability" not in dictionary_codes.columns:
        dictionary_codes["probability"] = None
    dictionary_codes = dictionary_codes[["document_id", "value", "dataset", "sampling_weight", "probability", "text"]]

    return dictionary_codes

def get_topic_codes(topic, codes):

    if is_null(codes):
        return None

    topic_codes = topic.model.apply_model(codes, probabilities=False)
    topic_codes["dataset"] = "topic_model"
    topic_codes["value"] = topic_codes[topic.name]
    topic_codes['probability'] = topic.model.apply_model(codes, probabilities=True)[topic.name]
    topic_codes["sampling_weight"] = topic_codes["sampling_weight"].astype(float)
    topic_codes["value"] = topic_codes["value"].astype(str)
    topic_codes = topic_codes[["document_id", "value", "dataset", "sampling_weight", "probability", "text"]]

    return topic_codes


def get_score(topic, df1, df2, sample_name, group_name):

    df = pandas.concat([df1, df2])
    score = compute_scores_from_dataset(df, "document_id", "value", "dataset", weight_column="sampling_weight")
    score = score[['coder1', 'coder1_mean', 'coder1_unweighted_mean', 'coder2', 'coder2_mean', 'coder2_unweighted_mean', 'n', 'pct_agree', 'accuracy', 'outcome_column', 'precision', 'recall', 'cohens_kappa', 'cohens_kappa_weighted', 'alpha']]
    score['topic'] = topic.name
    score['sample'] = sample_name
    score['group'] = group_name

    return score




class Command(BasicCommand):

    parameter_names = ["topic_model_name"]
    dependencies = []

    @staticmethod
    def add_arguments(parser):
        parser.add_argument("topic_model_name", type=str)
        parser.add_argument("--refresh_codes", action="store_true", default=False)
        parser.add_argument("--refresh_model", action="store_true", default=False)
        parser.add_argument("--skip_classifiers", action="store_true", default=False)
        parser.add_argument("--export", action="store_true", default=False)
        parser.add_argument("--num_cores", type=int, default=1)
        return parser

    @log_command
    def run(self):

        topic_model = get_model("TopicModel", app_name="django_learning").objects.get(name=self.parameters["topic_model_name"])
        topic_model.frame.get_sampling_flags(refresh=True)

        scores = None

        for topic in topic_model.topics.order_by("name"):

            print(topic)

            all_expert_codes = get_topic_expert_codes("expert_consensus", topic, "topic_model_{}".format(topic_model.name), [], exclude_consensus_ignore=True, refresh_codes=self.options["refresh_codes"])
            expert_codes = {}
            for coder in get_model("Project", app_name="django_learning").objects.get(name="topic_model_{}".format(topic_model.name)).coders.all():
                expert_codes[coder.name] = get_topic_expert_codes(coder.name, topic, "topic_model_{}".format(topic_model.name),
                                           [("filter_by_coder_names", [[coder.name]], {})],
                                           exclude_consensus_ignore=False, refresh_codes=self.options["refresh_codes"])

            dictionary_codes = get_topic_dictionary_codes(topic, all_expert_codes)
            topic_codes = get_topic_codes(topic, all_expert_codes)

            if is_not_null(all_expert_codes) and all_expert_codes['value'].astype(int).mean() < .05:
                warning = True
            else:
                warning = False

            for coder1, coder2 in itertools.combinations(expert_codes.keys(), 2):
                group_name = "{}_{}".format(coder1, coder2)
                df1 = expert_codes[coder1]
                df2 = expert_codes[coder2]
                if is_not_null(df1) and is_not_null(df2):
                    try: score = get_score(topic, df1, df2,  "topic_model_{}".format(topic_model.name), group_name)
                    except: score = None
                    if is_not_null(score):
                        score['warning'] = warning
                        if is_not_null(score):
                            if is_null(scores):
                                scores = score
                            else:
                                scores = pandas.concat([scores, score])
            for group_name, df1, df2 in [
                ("corex", all_expert_codes, topic_codes),
                ("dictionary", all_expert_codes, dictionary_codes)
            ]:

                if is_not_null(df1) and is_not_null(df2):
                    try: score = get_score(topic, df1, df2,  "topic_model_{}".format(topic_model.name), group_name)
                    except: score = None
                    if is_not_null(score):
                        score['warning'] = warning
                        if is_not_null(score):
                            if is_null(scores):
                                scores = score
                            else:
                                scores = pandas.concat([scores, score])

        if self.options["export"]:
            scores.to_csv("topic_model_{}_codes.csv".format(topic_model.name))

        return scores

