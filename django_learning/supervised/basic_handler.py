import importlib, copy, pandas
from abc import abstractmethod
from collections import OrderedDict

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss, make_scorer, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, FeatureUnion

from django_learning.utils import get_pipeline_repr, get_param_repr
from django_learning.utils.decorators import temp_cache_wrapper, require_model, require_training_data
from django_learning.utils.feature_extractors import BasicExtractor

from pewtils import is_null, is_not_null, recursive_update
from pewtils.django import CacheHandler


class SupervisedLearningHandler(object):

    pipeline_folders = []

    def __init__(self,
        cache_identifier,
        pipeline,
        outcome_variable=None,
        params=None,
        num_cores=2,
        verbose=True,
        **kwargs
    ):

        self.cache_identifier = cache_identifier
        self.outcome_variable = outcome_variable
        self.num_cores = num_cores
        self.pipeline_name = pipeline

        from django_learning.utils.pipelines import pipelines
        self._parameters = pipelines[pipeline]()

        if params:
            self._parameters = recursive_update(self._parameters, params)

        self.training_data = None
        self.model = None
        self.train_x = None
        self.train_y = None
        self.train_ids = None
        self.test_x = None
        self.test_y = None
        self.test_ids = None
        self.predict_y = None

        self.cache = CacheHandler("learning/supervised/{}".format(self.cache_identifier))
        self.temp_cache = CacheHandler("learning/feature_extractors/{}".format(self.cache_identifier), use_s3=False)

        if verbose:
            print "Initialized handler for {0}, {1}".format(self.cache_identifier, self.outcome_variable)


    @property
    def parameters(self):
        return self._parameters


    def load_training_data(self, refresh=False, only_load_existing=False, **kwargs):

        hashstr = "".join([
            self.cache_identifier,
            str(OrderedDict(sorted(self.parameters.get("codes", {}).items(), key=lambda t: t[0]))),
            str(OrderedDict(sorted(self.parameters.get("documents", {}).items(), key=lambda t: t[0])))
        ])

        cache_data = None

        if not refresh:
            cache_data = self.cache.read(hashstr)

        if is_null(cache_data) and not only_load_existing:

            cache_data = self._get_training_data(**kwargs)
            self.cache.write(hashstr, cache_data)

            self.model = None
            self.train_x = None
            self.train_y = None
            self.train_ids = None
            self.test_x = None
            self.test_y = None
            self.test_ids = None
            self.predict_y = None

        if is_not_null(cache_data):
            for k, v in cache_data.iteritems():
                setattr(self, k, v)


    @abstractmethod
    def _get_training_data(self):
        raise NotImplementedError


    @temp_cache_wrapper
    def load_model(self, refresh=False, clear_temp_cache=True, only_load_existing=False, **kwargs):

        pipeline_steps = copy.copy(self.parameters['pipeline']['steps'])
        params = self._collapse_pipeline_params(
            pipeline_steps,
            self.parameters['pipeline']['params']
        )

        if "name" in self.parameters["model"].keys():

            if "classification" in self.pipeline_folders and "regression" not in str(self.__class__):
                model_module = importlib.import_module(
                    "django_learning.supervised.models.classification.{0}".format(self.parameters["model"]['name'])
                )
            elif "regression" in self.pipeline_folders or "regression" in str(self.__class__):
                model_module = importlib.import_module(
                    "django_learning.supervised.models.regression.{0}".format(self.parameters["model"]['name'])
                )
            else:
                model_module = importlib.import_module(
                    "django_learning.supervised.models.{0}".format(self.parameters["model"]['name'])
                )

            model_params = model_module.get_params()
            model_class = model_params.pop("model_class")
            model_params = model_params["params"]
            pipeline_steps.append(("model", model_class))

            params.update({"model__{}".format(k): v for k, v in model_params.iteritems()})
            if 'params' in self.parameters['model'].keys():
                params.update({"model__{}".format(k): v for k, v in self.parameters['model']['params'].iteritems()})

        hashstr = "".join([
            self.cache_identifier,
            str(OrderedDict(sorted(self.parameters.get("codes", {}).items(), key=lambda t: t[0]))),
            str(OrderedDict(sorted(self.parameters.get("documents", {}).items(), key=lambda t: t[0]))),
            str(get_pipeline_repr(pipeline_steps)),
            str(get_param_repr(params)),
            str(OrderedDict(sorted(self.parameters.get("model", {}).items(), key=lambda t: t[0])))
        ])

        cache_data = None

        if not refresh:

            cache_data = self.cache.read(hashstr)

            # cache_data['test_y'] = cache_data['test_y'].reset_index()
            # del cache_data['test_y']['index']
            # cache_data['test_x'] = cache_data['test_x'].reset_index()
            # del cache_data['test_x']['index']
            # cache_data['test_ids'] = cache_data['test_x'].index
            # print "Resetting test indices"
            # self.cache.write(hashstr, cache_data)

            # if "test_x_old" not in cache_data.keys():
            #     from logos.models import *
            #     cache_data['test_x_old'] = cache_data['test_x']
            #     cache_data['test_y_old'] = cache_data['test_y']
            #     cache_data['test_ids_old'] = cache_data['test_ids']
            #     cache_data['predict_y_old'] = cache_data['predict_y']
            #     good_docs = DocumentSampleDocument.objects.filter(sample__in=DocumentSample.objects.filter(pk__in=[12, 22])).values_list("document_id", flat=True)
            #     print "Old length: {}".format(len(cache_data['test_x']))
            #     cache_data['test_x'] = cache_data['test_x'][cache_data['test_x']['document_id'].isin(good_docs)]
            #     print "New length: {}".format(len(cache_data['test_x']))
            #     cache_data['test_y'] = cache_data['test_y'].iloc[cache_data['test_x'].index]
            #     cache_data['test_ids'] = cache_data['test_x'].index
            #     cache_data['predict_y'] = pandas.Series(cache_data['predict_y']).iloc[cache_data['test_x'].index].values
            #     self.cache.write(hashstr, cache_data)


        if is_null(cache_data) and not only_load_existing:

            cache_data = CacheHandler("learning/supervised").read(hashstr)
            if is_null(cache_data):
                cache_data = self._get_model(pipeline_steps, params , **kwargs)
            self.cache.write(hashstr, cache_data)

        if is_not_null(cache_data):
            for k, v in cache_data.iteritems():
                setattr(self, k, v)


    @require_training_data
    def _get_model(self, pipeline_steps, params , **kwargs):

        df = self.training_data

        smallest_code = df[self.outcome_variable].value_counts(ascending=True).index[0]
        largest_code = df[self.outcome_variable].value_counts(ascending=False).index[0]

        # print "Code frequencies: {}".format(dict(df[self.outcome_variable].value_counts(ascending=True)))

        if "training_weight" not in df.columns:
            df["training_weight"] = 1.0

        if self.parameters["model"].get("use_class_weights", False):

            class_weights = {}
            base_weight = df[df[self.outcome_variable]==largest_code]['training_weight'].sum()
            # total_weight = df['training_weight'].sum()
            for c in df[self.outcome_variable].unique():
                # class_weights[c] = float(df[df[self.outcome_variable]==c]["training_weight"].sum()) / float(total_weight)
                class_weights[c] = base_weight / float(df[df[self.outcome_variable]==c]['training_weight'].sum())
            total_weight = sum(class_weights.values())
            class_weights = {k: float(v)/float(total_weight) for k, v in class_weights.items()}
            params["model__class_weight"] = [class_weights, ]
            print "Class weights: {}".format(class_weights)

        print "Creating train-test split"

        y = df[self.outcome_variable]
        X_cols = df.columns.tolist()
        X_cols.remove(self.outcome_variable)
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

        scoring_function = None
        if "scoring_function" in self.parameters["model"].keys():
            scoring_function = self._get_scoring_function(
                self.parameters["model"]["scoring_function"],
                binary_base_code=smallest_code if len(y.unique()) == 2 else None
            )

        print "Beginning grid search using %s and %i cores for %s" % (
            str(scoring_function),
            self.num_cores,
            self.outcome_variable
        )

        model = GridSearchCV(
            Pipeline(pipeline_steps),
            params,
            fit_params={'model__sample_weight': [x for x in X_train["training_weight"].values]} if self.parameters["model"].get("use_sample_weights", False) else {},
            cv=self.parameters["model"].get("cv", 5),
            n_jobs=self.num_cores,
            verbose=1,
            scoring=scoring_function
        )

        model.fit(X_train, y_train)

        print "Finished training model, best score: {}".format(model.best_score_)

        predict_y = model.predict(X_test) if is_not_null(X_test) else None

        cache_data = {
            "model": model,
            "train_x": X_train,
            "train_y": y_train,
            "train_ids": train_ids,
            "test_x": X_test,
            "test_y": y_test,
            "test_ids": test_ids,
            "predict_y": predict_y
        }

        return cache_data

    def _get_scoring_function(self, func_name, binary_base_code=None):

        try:

            from django_learning.utils.scoring_functions import scoring_functions
            scoring_function = scoring_functions[func_name]
            # scoring_function = importlib.import_module(
            #     "logos.learning.utils.scoring_functions.{0}".format(func_name)
            # )
            scoring_function = make_scorer(scoring_function.scorer)

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
                final_params.update(self._collapse_pipeline_params(step, params, names=names+[sname]))
        elif isinstance(pipeline, FeatureUnion):
            final_params.update(self._collapse_pipeline_params(pipeline.transformer_list, params, names=names))
        elif isinstance(pipeline, tuple):
            final_params.update(pipeline[1], params, names=names+[pipeline[0]])
        elif isinstance(pipeline, list):
            for sname, step in pipeline:
                final_params.update(self._collapse_pipeline_params(step, params, names=names+[sname]))
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
    def print_report(self):

        print "'%s' results" % self.outcome_variable

        print "Best score: {} ({} std.)".format(self.model.best_score_, getattr(self.model, "best_score_std_", None))

        # print "Best parameters:"
        # params = self.model.best_params_
        # for p in params.keys():
        #     if p.endswith("__stop_words"):
        #         del params[p]
        # print params

        try: self.show_top_features()
        except: pass


    @require_model
    @temp_cache_wrapper
    def apply_model(self, data, keep_cols=None, clear_temp_cache=True):

        if not keep_cols: keep_cols = []

        predictions = self.model.predict(data)
        try: probabilities = self.model.predict_proba(data)
        except AttributeError: probabilities = [None] * len(data)

        codes = []
        for index, pred, prob in zip(data.index, predictions, probabilities):
            if type(prob) == list or type(prob) == tuple:
                prob = max(prob)
            code = {
                self.outcome_variable: pred,
                "probability": prob
            }
            for col in keep_cols:
                code[col] = data.loc[index, col]
            codes.append(code)

        return pandas.DataFrame(codes)