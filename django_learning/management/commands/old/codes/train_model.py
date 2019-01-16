import pandas
from django.core.management.base import BaseCommand, CommandError
from django.db.models import Count

from logos.models import CodeVariable, CodeVariableClassifier
from logos.utils import clean_text, get_congress_stopwords, get_model_by_document_type
from pewtils.internal import agg_any

from sklearn.metrics import f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn import metrics, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


class Command(BaseCommand):
    """
    """
    help = ""

    def add_arguments(self, parser):

        parser.add_argument("--code_variable", default=None, type=str)
        parser.add_argument("--ignore_mturk",  default=False, action="store_true")
        parser.add_argument("--mturk_only",  default=False, action="store_true")
        parser.add_argument("--consolidate_mturk_codes",  default=False, action="store_true")
        parser.add_argument("--model_type", default="linear_svc", type=str)
        parser.add_argument("--scoring_metric", default="precision", type=str)
        parser.add_argument("--num_cores", default=2, type=int)
        parser.add_argument("--flag_code_if_any", default=False, action="store_true")
        parser.add_argument("--test_percent", default=.25, type=float)
        parser.add_argument("--min_year", default=2013, type=int)
        parser.add_argument("--document_type", default="press_releases", type=str)

    def handle(self, *args, **options):

        # doc_model = get_model_by_document_type(options["document_type"])
        # metadata = doc_model.objects.metadata()
        doc_model, metadata = get_model_by_document_type(options["document_type"])

        if options["code_variable"] == "all":
            # code_vars = CodeVariable.objects.all()
            code_vars_names = metadata["coder_code_model"].objects.available_codes()
            code_vars = CodeVariable.objects.get(name__in=code_vars_names)
        else:
            code_vars = [CodeVariable.objects.get(name=options["code_variable"])]

        for code_variable in code_vars:

            print "Selecting %s for %s" % (metadata["name_plural"], code_variable.name)
            # if options["ignore_mturk"]:
            #     prs = list(metadata["coder_code_model"].objects.filter(code__variable=code_variable).filter(coder__is_mturk=False).values())
            # elif options["mturk_only"]:
            #     prs = list(metadata["coder_code_model"].objects.filter(code__variable=code_variable).filter(coder__is_mturk=True).values())
            # elif options["consolidate_mturk_codes"]:
            #     prs = list(metadata["coder_code_model"].objects.filter(code__variable=code_variable).filter(coder__is_mturk=False).values())
            #     df = pandas.DataFrame(list(metadata["coder_code_model"].objects.filter(code__variable=code_variable).filter(coder__is_mturk=True).values()))
            #     if options["flag_code_if_any"]:
            #         df = df.groupby(metadata["id_field"]).agg(agg_any)
            #     else:
            #         df = df.groupby(metadata["id_field"]).agg(lambda x: x.value_counts().index[0])
            #     for index, row in df.iterrows():
            #         prs.append({
            #             "%s" % metadata["id_field"]: index,
            #             "coder_id": "mturk",
            #             "code_id": row["code_id"]
            #         })
            # else:
            #     prs = list(metadata["coder_code_model"].objects.filter(code__variable=code_variable).values())
            # df = pandas.DataFrame(prs)
            # if options["flag_code_if_any"]:
            #     df = df.groupby(metadata["id_field"]).agg(agg_any)
            # else:
            #     df = df.groupby(metadata["id_field"]).agg(lambda x: x.value_counts().index[0])
            # ids = df.index
            # df = df.merge(pandas.DataFrame(list(doc_model.objects.filter(pk__in=ids).values("pk", metadata["text_field"]))), how='left', left_index=True, right_on="pk")
            # df.index = ids

            df = metadata["coder_code_model"].objects.classification_data(code_variable=code_variable, coder_type='all', consolidate_type=None)

            print "Selected %i %s for %s" % (len(df[metadata["text_field"]]), metadata["name_plural"], code_variable.name)

            print "Creating train-test split for %s %s" % (code_variable.name, metadata["name_plural"])
            X = df[metadata["text_field"]]
            # y = df['code_id']
            y = df["code__value"]
            X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(X, y, y.index, test_size=options["test_percent"], random_state=5)
            print "Selected %i training cases and %i test cases for %s %s" % (len(y_train), len(y_test), code_variable.name, metadata["id_field"])

            params = {
                'vec__sublinear_tf': (False, ),
                'vec__max_df': (0.6, 0.75, ),
                'vec__min_df': (5, ),
                'vec__max_features': (None, ) ,
                'vec__ngram_range': ((1, 3), ),
                'vec__use_idf': (True, ),
                'vec__norm': ('l1', 'l2'),
                'vec__stop_words': (get_congress_stopwords(add_english=True), ),
                # 'vec__stop_words': ('english', get_congress_stopwords(add_english=True), ),
                'vec__preprocessor': (clean_text, ),
                # 'vec__preprocessor': (None, filter_nouns),
            }

            model_params = {
                "linear_svc": {
                    'clf__max_iter': (1000, ),
                    'clf__penalty': ('l2', ),
                    'clf__class_weight' : (None, 'auto', ),
                    # 'clf__loss': ('squared_hinge', 'hinge', )
                    'clf__loss': ('hinge', )
                },
                "sgd": {
                    # 'clf__loss': ('hinge', 'squared_hinge', 'log', 'modified_huber', 'perceptron', ),
                    'clf__loss': ('hinge', 'log', 'modified_huber', ),
                    'clf__penalty': ('elasticnet', ),
                    'clf__l1_ratio': (.05, .15, .25, ),
                    'clf__alpha': (.0001, ),
                    'clf__n_iter': (5, ),
                    'clf__class_weight': ('auto', None, ),
                    # 'clf__average': (True, False, )
                    'clf__average': (True, )
                },
                "knn": {
                    'clf__weights': ('uniform', 'distance', ),
                    'clf__algorithm': ('auto', 'ball_tree', 'kd_tree', 'brute', ),
                    'clf__leaf_size': (30, 50, 70, )
                },
                'gradient_boosting': {
                    'clf__loss': ('deviance', 'exponential', ),
                    'clf__learning_rate': (.05, .1, .2, ),
                    'clf__n_estimators': (100, ),
                    'clf__max_depth': (3, ),
                    'clf__min_samples_split': (2, 5, ),
                    'clf__min_samples_leaf': (1, 2, ),
                    'clf__max_features': ('auto', 'sqrt', 'log2', None, )
                }
            }

            model_pipelines = {
                "linear_svc": Pipeline([
                    ("vec", TfidfVectorizer()),
                    ("trans", DenseTransformer()),
                    ("clf", LinearSVC())
                ]),
                "sgd": Pipeline([
                    ("vec", TfidfVectorizer()),
                    ("trans", DenseTransformer()),
                    ("clf", SGDClassifier())
                ]),
                "knn": Pipeline([
                    ("vec", TfidfVectorizer()),
                    ("trans", DenseTransformer()),
                    ("clf", KNeighborsClassifier())
                ]),
                "gradient_boosting": Pipeline([
                    ("vec", TfidfVectorizer()),
                    ("trans", DenseTransformer()),
                    ("clf", GradientBoostingClassifier())
                ])
            }

            if code_variable.codes.count() == 2:
                smallest_code = code_variable.codes.annotate(count=Count("coder_%s" % metadata["name_plural"])).order_by("count")[0]
                scoring_function = make_scorer(f1_score, pos_label=smallest_code.pk) # equivalent to f1_binary
            else:
                scoring_function = "f1_macro"
                # scoring_function = "f1_micro"
                # scoring_function = "f1_weighted"

            print "Beginning %s grid search using %s and %i cores for %s" % (options["model_type"], str(scoring_function), options["num_cores"], code_variable.name)
            params.update(model_params[options["model_type"]])
            clf = GridSearchCV(
                model_pipelines[options["model_type"]],
                params,
                cv=5,
                scoring=scoring_function,
                n_jobs=options["num_cores"],
                verbose=1
            )
            clf.fit(X_train, y_train)

            if code_variable.model:
                CodeVariableClassifier.objects.get(pk=code_variable.model.pk).delete()
            model = CodeVariableClassifier(
                parameters = params,
                model = clf,
                training_data = (X_train, y_train, train_ids),
                testing_data = (X_test, y_test, test_ids),
                document_type = options["document_type"]
            )
            model.save()
            code_variable.model = model
            code_variable.save()

            code_variable.model.print_report()
            code_variable.model.print_top_features()

            print "Model saved"


# def filter_nouns(text, filter_pos='NN'):
#
#     text = text.split()
#     tagged_words = nltk.pos_tag(text)
#     non_nouns= [word[0] for word in tagged_words if word[1] != filter_pos]
#     if len(non_nouns) == len(text):
#         print(tagged_words)
#     elif len(non_nouns) < 1:
#         print("short: %s".format(text))
#     return " ".join(non_nouns)

from sklearn.base import TransformerMixin, BaseEstimator
## Must inherit from both to be used in Grid Search

class DenseTransformer(TransformerMixin, BaseEstimator):
    """ Transforms a sparse matrix into dense format
        Some models require this step
    """

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self