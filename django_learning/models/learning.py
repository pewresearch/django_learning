import importlib, copy, pandas, numpy, os

from django.db import models
from django.db.models import Q
from django.contrib.postgres.fields import ArrayField
from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType

from picklefield.fields import PickledObjectField
from langdetect import detect
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from statsmodels.stats.inter_rater import cohens_kappa

from sklearn.cross_validation import train_test_split
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss, make_scorer, mean_squared_error, r2_score, matthews_corrcoef, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind

from django_commander.models import LoggedExtendedModel

from django_learning.settings import S3_CACHE_PATH, LOCAL_CACHE_PATH
from django_learning.utils import get_document_types, get_pipeline_repr, get_param_repr
from django_learning.utils.pipelines import pipelines
from django_learning.utils.dataset_extractors import dataset_extractors
from django_learning.utils.decorators import require_training_data, require_model, temp_cache_wrapper
from django_learning.utils.feature_extractors import BasicExtractor
from django_learning.utils.models import models as learning_models
from django_learning.utils.scoring import compute_scores_from_datasets_as_coders
from django_learning.utils.scoring_functions import scoring_functions

from pewtils import is_not_null, is_null, decode_text, recursive_update
from pewtils.django import get_model, CacheHandler
from pewtils.sampling import compute_sample_weights_from_frame, compute_balanced_sample_weights
from pewtils.stats import wmom


class LearningModel(LoggedExtendedModel):

    project = models.ForeignKey("django_learning.Project", related_name="+", null=True)
    name = models.CharField(max_length=100, unique=True, help_text="Unique name of the classifier")

    pipeline_name = models.CharField(max_length=150, null=True, help_text="The named pipeline used to seed the handler's parameters, if any; note that the JSON pipeline file may have changed since this classifier was created; refer to the parameters field to view the exact parameters used to compute the model")
    parameters = PickledObjectField(null=True, help_text="A pickle file of the parameters used to process the codes and generate the model")

    cv_folds = PickledObjectField(null=True)
    cv_folds_test = PickledObjectField(null=True)

    cache_hash =  models.CharField(max_length=256, null=True)

    num_cores = models.IntegerField(default=1)

    class Meta:

        abstract = True

    def __str__(self):
        return self.name

    def __init__(self, *args, **kwargs):

        super(LearningModel, self).__init__(*args, **kwargs)

        params = {}
        if self.pipeline_name:
            params.update(pipelines[self.pipeline_name]())
        try: self.parameters = recursive_update(params, self.parameters if self.parameters else {})
        except AttributeError:
            print "WARNING: couldn't update parameters from pipeline, it may not exist anymore!"

        self.cache_identifier = "{}_{}".format(self.name, self.pipeline_name)

        try:
            self.dataset_extractor = dataset_extractors[self.parameters["dataset_extractor"]["name"]](
                **self.parameters["dataset_extractor"]["parameters"]
            )
        except TypeError:
            print "WARNING: couldn't identify dataset extractor, it may not exist anymore!"
        self.dataset = None

        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.predict_dataset = None
        # self.X = None
        # self.Y = None
        # self.train_x = None
        # self.train_y = None
        # self.train_ids = None
        # self.test_x = None
        # self.test_y = None
        # self.test_ids = None
        # self.predict_y = None

        self.cache = CacheHandler(os.path.join(S3_CACHE_PATH, "learning_models/{}".format(self.cache_identifier)),
            hash = False,
            use_s3=True,
            aws_access=settings.AWS_ACCESS_KEY_ID,
            aws_secret=settings.AWS_SECRET_ACCESS_KEY,
            bucket=settings.S3_BUCKET
        )
        self.temp_cache = CacheHandler(os.path.join(LOCAL_CACHE_PATH, "feature_extractors/{}".format(self.cache_identifier)),
            hash=False,
            use_s3=False
        )

    def save(self, *args, **kwargs):

        # params = {}
        # if self.pipeline_name:
        #     params.update(pipelines[self.pipeline_name]())
        # self.parameters = recursive_update(params, self.parameters if self.parameters else {})
        #
        # self.parameters['pipeline']['steps'] = [(k, v) for k, v in self.parameters['pipeline']['steps'] if k != 'model']
        # # TODO: figure out where this bug is occuring, but for now we'll deal with extra "model" params sneaking in

        super(LearningModel, self).save(*args, **kwargs)

    def extract_dataset(self, refresh=False, **kwargs):

        self.dataset = self.dataset_extractor.extract(refresh=refresh, **kwargs)
        if hasattr(self.dataset_extractor, "project") and self.dataset_extractor.project:
            self.project = get_model("Project", app_name="django_learning").objects.get(name=self.dataset_extractor.project.name)
            self.save()
        if not self.dataset_extractor.outcome_column:
            outcome_col = self.parameters["dataset_extractor"].get("outcome_column", None)
            if outcome_col:
                self.dataset_extractor.set_outcome_column(outcome_col)
            else:
                raise Exception("Extractor '{}' has no outcome column set and one was not specified in your pipeline".format(
                    self.parameters["dataset_extractor"]["name"]
                ))
        self.outcome_column = self.dataset_extractor.outcome_column

    @temp_cache_wrapper
    def load_model(self, refresh=False, clear_temp_cache=True, only_load_existing=False, **kwargs):

        if is_null(self.dataset):
            self.extract_dataset()

        cache_data = None

        if not refresh and self.cache_hash:
            cache_data = self.cache.read(self.cache_hash)
            # note: if you hit an ImportError when loading the pickle, you probably need to
            # scroll up to the top and load a utils module in this file
            # (e.g. from django_learning.utils.scoring_functions import scoring_functions)

        if is_null(cache_data) and not only_load_existing:

            pipeline_steps = copy.copy(self.parameters['pipeline']['steps'])
            params = self._collapse_pipeline_params(
                pipeline_steps,
                self.parameters['pipeline']['params']
            )

            if "name" in self.parameters["model"].keys():

                model_params = learning_models[self.parameters["model"]["name"]]()
                model_class = model_params.pop("model_class")
                model_params = model_params["params"]
                pipeline_steps.append(("model", model_class))

                params.update({"model__{}".format(k): v for k, v in model_params.iteritems()})
                if 'params' in self.parameters['model'].keys():
                    params.update({"model__{}".format(k): v for k, v in self.parameters['model']['params'].iteritems()})

            updated_hashstr = "".join([
                self.cache_identifier,
                self.dataset_extractor.get_hash(),
                str(get_pipeline_repr(pipeline_steps)),
                str(get_param_repr(params)),
                str(OrderedDict(sorted(self.parameters.get("model", {}).items(), key=lambda t: t[0])))
            ])
            updated_hashstr = self.cache.file_handler.get_key_hash(updated_hashstr)
            cache_data = self._train_model(pipeline_steps, params, **kwargs)
            self.cache.write(updated_hashstr, cache_data)
            self.cache_hash = updated_hashstr
            self.cv_folds = None
            self.save()

        if is_not_null(cache_data):
            for k, v in cache_data.iteritems():
                setattr(self, k, v)

    def _train_model(self, pipeline_steps, params, **kwargs):

        df = self.dataset

        smallest_code = df[self.dataset_extractor.outcome_column].value_counts(ascending=True).index[0]
        largest_code = df[self.dataset_extractor.outcome_column].value_counts(ascending=False).index[0]

        # print "Code frequencies: {}".format(dict(df[self.outcome_variable].value_counts(ascending=True)))

        if "training_weight" not in df.columns:
            df["training_weight"] = 1.0

        if self.parameters["model"].get("use_class_weights", False):

            class_weights = {}
            base_weight = df[df[self.dataset_extractor.outcome_column] == largest_code]['training_weight'].sum()
            # total_weight = df['training_weight'].sum()
            for c in df[self.dataset_extractor.outcome_column].unique():
                # class_weights[c] = float(df[df[self.outcome_column]==c]["training_weight"].sum()) / float(total_weight)
                class_weights[c] = base_weight / float(df[df[self.dataset_extractor.outcome_column] == c]['training_weight'].sum())
            total_weight = sum(class_weights.values())
            class_weights = {k: float(v) / float(total_weight) for k, v in class_weights.items()}
            params["model__class_weight"] = [class_weights, ]
            print "Class weights: {}".format(class_weights)

        print "Creating train-test split"

        y = df[self.dataset_extractor.outcome_column]
        X_cols = df.columns.tolist()
        X_cols.remove(self.dataset_extractor.outcome_column)
        X = df[X_cols]
        if self.parameters["model"]["test_percent"] == 0.0:
            X_train, X_test, y_train, y_test, train_ids, test_ids = X, None, y, None, y.index, None
            print "Training on all {} cases".format(len(y_train))
        else:
            X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(X, y, y.index, test_size=self.parameters["model"]["test_percent"], random_state=5)
            print "Selected %i training cases and %i test cases" % (
                len(y_train),
                len(y_test)
            )

        train_dataset = df.ix[train_ids]
        test_dataset = df.ix[test_ids] if len(test_ids) > 0 else None

        scoring_function = None
        if "scoring_function" in self.parameters["model"].keys():
            scoring_function = self._get_scoring_function(
                self.parameters["model"]["scoring_function"],
                binary_base_code=smallest_code if len(y.unique()) == 2 else None
            )

        print "Beginning grid search using %s and %i cores for %s" % (
            str(scoring_function),
            self.num_cores,
            self.dataset_extractor.outcome_column
        )

        model = GridSearchCV(
            Pipeline(pipeline_steps),
            params,
            fit_params={'model__sample_weight': [x for x in train_dataset["training_weight"].values]} if self.parameters["model"].get("use_sample_weights", False) else {},
            cv=self.parameters["model"].get("cv", 5),
            n_jobs=self.num_cores,
            verbose=1,
            scoring=scoring_function
        )

        model.fit(train_dataset, train_dataset[self.dataset_extractor.outcome_column])

        print "Finished training model, best score: {}".format(model.best_score_)

        cache_data = {
            "model": model,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            # "predict_dataset": predict_dataset
        }

        return cache_data

    @require_model
    def describe_model(self):

        print "'{}' results".format(self.dataset_extractor.outcome_column)

        print "Best score: {} ({} std.)".format(self.model.best_score_,
                                                getattr(self.model, "best_score_std_", None))

        # print "Best parameters:"
        # params = self.model.best_params_
        # for p in params.keys():
        #     if p.endswith("__stop_words"):
        #         del params[p]
        # print params

    @require_model
    def get_test_prediction_results(self, refresh=False):

        self.predict_dataset = None
        if is_not_null(self.test_dataset):
            self.predict_dataset = self.produce_prediction_dataset(self.test_dataset, cache_key="predict_main", refresh=refresh)
            scores = self.compute_prediction_scores(self.test_dataset, cache_key="predict_main", refresh=False)
            return scores

    def print_test_prediction_report(self):

        print self.get_test_prediction_results()

    @require_model
    def get_cv_prediction_results(self, refresh=False):

        print "Computing cross-fold predictions"
        _final_model = self.model
        _final_model_best_estimator = self.model.best_estimator_
        all_fold_scores = []
        if not refresh and self.cv_folds:
            cv_folds = self.cv_folds
        else:
            self.cv_folds = [f for f in KFold(n_splits=self.parameters["model"].get("cv", 5), shuffle=True).split(self.dataset.index)]
            self.save()
        for i, folds in enumerate(self.cv_folds):
            fold_train_index, fold_test_index = folds
            fold_train_dataset = self.dataset.ix[fold_train_index]
            fold_test_dataset = self.dataset.ix[fold_test_index]
            self.model = _final_model_best_estimator.fit(fold_train_dataset, fold_train_dataset[self.dataset_extractor.outcome_column])
            fold_predict_dataset = self.produce_prediction_dataset(fold_test_dataset, cache_key="predict_fold_{}".format(i), refresh=refresh)
            fold_scores = self.compute_prediction_scores(fold_test_dataset, cache_key="predict_fold_{}".format(i), refresh=False)
            all_fold_scores.append(fold_scores)
        self.model = _final_model
        fold_score_df = pandas.concat(all_fold_scores)
        return pandas.concat([
            all_fold_scores[0][["coder1", "coder2", "outcome_column"]],
            fold_score_df.groupby(fold_score_df.index).mean()
        ], axis=1)

    def print_cv_prediction_report(self):

        print self.get_cv_prediction_results()

    def _get_scoring_function(self, func_name, binary_base_code=None):

        try:

            from django_learning.utils.scoring_functions import scoring_functions
            scoring_function = make_scorer(scoring_functions[func_name])

        except:

            if "regression" in str(self.__class__):
                func_map = {
                    "mean_squared_error": (mean_squared_error, False, False),
                    "r2": (r2_score, True, False)
                }
                func, direction, needs_proba = func_map[func_name]
                scoring_function = make_scorer(func, needs_proba=needs_proba, greater_is_better=direction)
            elif binary_base_code:
                func_map = {
                    "f1": (f1_score, True, False),
                    "precision": (precision_score, True, False),
                    "recall": (recall_score, True, False),
                    "brier_loss": (brier_score_loss, False, True)
                }
                func, direction, needs_proba = func_map[func_name]
                scoring_function = make_scorer(func, needs_proba=needs_proba, greater_is_better=direction,
                                               pos_label=binary_base_code)
            else:
                if self.parameters["model"]["scoring_function"] == "f1":
                    scoring_function = "f1_macro"
                    # scoring_function = "f1_micro"
                    # scoring_function = "f1_weighted"
                elif self.parameters["model"]["scoring_function"] == "precision":
                    scoring_function = "precision"
                else:
                    scoring_function = "recall"

        return scoring_function

    def _collapse_pipeline_params(self, pipeline, params, names=None):

        final_params = {}
        if not names:
            names = []
        if isinstance(pipeline, Pipeline):
            for sname, step in pipeline.steps:
                final_params.update(self._collapse_pipeline_params(step, params, names=names + [sname]))
        elif isinstance(pipeline, FeatureUnion):
            final_params.update(self._collapse_pipeline_params(pipeline.transformer_list, params, names=names))
        elif isinstance(pipeline, tuple):
            final_params.update(pipeline[1], params, names=names + [pipeline[0]])
        elif isinstance(pipeline, list):
            for sname, step in pipeline:
                final_params.update(self._collapse_pipeline_params(step, params, names=names + [sname]))
        else:
            if names[-1] in params.keys():
                for k, v in params[names[-1]].iteritems():
                    # if k == "preprocessors":
                    #     preprocessor_sets = []
                    #     for pset in v:
                    #         preprocessors = []
                    #         try:
                    #             for preprocessor_name, preprocessor_params in pset:
                    #                 preprocessor_module = importlib.import_module("logos.learning.utils.preprocessors.{0}".format(preprocessor_name))
                    #                 preprocessors.append(preprocessor_module.Preprocessor(**preprocessor_params))
                    #         except ValueError: pass
                    #         preprocessor_sets.append(preprocessors)
                    #     v = preprocessor_sets
                    if len(v) > 0:
                        final_params["__".join(names + [k])] = v
            if isinstance(pipeline, BasicExtractor):
                final_params["{}__cache_identifier".format("__".join(names))] = [self.cache_identifier]
                final_params["{}__feature_name_prefix".format("__".join(names))] = [names[-1]]
                if hasattr(self, 'document_types'):
                    final_params["{}__document_types".format("__".join(names))] = [self.document_types]

        return final_params

    @require_model
    def get_feature_names(self, m):

        features = []

        if hasattr(m, "steps"):
            for name, step in m.steps:
                features.append(self.get_feature_names(step))
        elif hasattr(m, "transformer_list"):
            for name, step in m.transformer_list:
                features.append(self.get_feature_names(step))
        elif hasattr(m, "get_feature_names"):
            return m.get_feature_names()

        return [f for sublist in features for f in sublist]

    @require_model
    @temp_cache_wrapper
    def apply_model(self, data, keep_cols=None, clear_temp_cache=True):

        if not keep_cols: keep_cols = []

        predictions = self.model.predict(data)
        try:
            probabilities = self.model.predict_proba(data)
        except AttributeError:
            probabilities = [None] * len(data)

        labels = []
        for index, pred, prob in zip(data.index, predictions, probabilities):
            if type(prob) in [list, tuple, numpy.ndarray]:
                prob = max(prob)
            label = {
                self.dataset_extractor.outcome_column: pred,
                "probability": prob
            }
            for col in keep_cols:
                label[col] = data.loc[index, col]
            labels.append(label)

        return pandas.DataFrame(labels, index=data.index)

    @require_model
    def produce_prediction_dataset(self, df_to_predict, cache_key=None, refresh=False):

        predicted_df = dataset_extractors["model_prediction_dataset"](dataset=df_to_predict, learning_model=self, cache_key=cache_key).extract(refresh=refresh)
        return predicted_df

    @require_model
    def compute_prediction_scores(self, df_to_predict, cache_key=None, refresh=False):

        if "sampling_weight" in df_to_predict.columns: weight_col = "sampling_weight"
        else: weight_col = None
        predicted_df = self.produce_prediction_dataset(df_to_predict, cache_key=cache_key, refresh=refresh)
        return compute_scores_from_datasets_as_coders(df_to_predict, predicted_df, "index", self.dataset_extractor.outcome_column, weight_column=weight_col)


class DocumentLearningModel(LearningModel):

    sampling_frame = models.ForeignKey("django_learning.SamplingFrame", related_name="learning_models")
    # question = models.ForeignKey("django_learning.Question", related_name="learning_models")

    # INHERITED FIELDS
    # name = models.CharField(max_length=100, unique=True, help_text="Unique name of the classifier")
    # outcome_column = models.CharField(max_length=256)
    # pipeline_name = models.CharField(max_length=150, null=True,
    #                                  help_text="The named pipeline used to seed the handler's parameters, if any; note that the JSON pipeline file may have changed since this classifier was created; refer to the parameters field to view the exact parameters used to compute the model")
    # parameters = PickledObjectField(null=True,
    #                                 help_text="A pickle file of the parameters used to process the codes and generate the model")
    #
    # cv_folds = PickledObjectField(null=True)
    # cv_folds_test = PickledObjectField(null=True)
    #
    # training_data_hash = models.CharField(max_length=256, null=True)
    # model_hash = models.CharField(max_length=256, null=True)

    class Meta:

        abstract = True

    def extract_dataset(self, refresh=False):

        super(DocumentLearningModel, self).extract_dataset(refresh=refresh)
        if hasattr(self.dataset_extractor, "sampling_frame") and self.dataset_extractor.sampling_frame:
            self.sampling_frame = get_model("SamplingFrame", app_name="django_learning").objects.get(name=self.dataset_extractor.sampling_frame.name)
            self.save()


