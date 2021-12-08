from __future__ import print_function
import importlib, copy, pandas, numpy, os

from django.db import models
from django.conf import settings

from picklefield.fields import PickledObjectField
from collections import OrderedDict
from tempfile import mkdtemp
from shutil import rmtree

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    make_scorer,
    mean_squared_error,
    r2_score,
    matthews_corrcoef,
    accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import ParameterGrid

from django_commander.models import LoggedExtendedModel

from django_learning.utils import get_pipeline_repr, get_param_repr
from django_learning.utils.decorators import require_model, temp_cache_wrapper
from django_learning.utils.scoring import compute_scores_from_datasets_as_coders

from pewtils import is_not_null, is_null, recursive_update
from django_pewtils import get_model, CacheHandler


class LearningModel(LoggedExtendedModel):

    project = models.ForeignKey(
        "django_learning.Project",
        related_name="+",
        null=True,
        on_delete=models.SET_NULL,
    )
    name = models.CharField(
        max_length=100, unique=True, help_text="Unique name of the classifier"
    )

    pipeline_name = models.CharField(
        max_length=150,
        null=True,
        help_text="The named pipeline used to seed the handler's parameters, if any; note that the JSON pipeline file may have changed since this classifier was created; refer to the parameters field to view the exact parameters used to compute the model",
    )
    parameters = PickledObjectField(
        null=True,
        help_text="A pickle file of the parameters used to process the codes and generate the model",
    )

    cv_folds = PickledObjectField(null=True)
    cv_folds_test = PickledObjectField(null=True)

    model_cache_hash = models.CharField(max_length=256, null=True)
    dataset_cache_hash = models.CharField(max_length=256, null=True)
    test_dataset_cache_hash = models.CharField(max_length=256, null=True)

    class Meta:

        abstract = True

    def __str__(self):
        return self.name

    def __init__(self, *args, **kwargs):

        super(LearningModel, self).__init__(*args, **kwargs)

        self.cache_identifier = "{}_{}".format(self.name, self.pipeline_name)

        # self.dataset_extractor = self._get_dataset_extractor("dataset_extractor")
        self.dataset = None

        self.model = None
        self.train_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.cache = CacheHandler(
            os.path.join(
                settings.DJANGO_LEARNING_S3_CACHE_PATH,
                "learning_models/{}".format(self.cache_identifier),
            ),
            hash=False,
            use_s3=settings.DJANGO_LEARNING_USE_S3,
            bucket=settings.S3_BUCKET,
        )
        self.dataset_cache = CacheHandler(
            os.path.join(settings.DJANGO_LEARNING_S3_CACHE_PATH, "datasets"),
            hash=False,
            use_s3=settings.DJANGO_LEARNING_USE_S3,
            bucket=settings.S3_BUCKET,
        )
        self.temp_cache = CacheHandler(
            os.path.join(
                settings.DJANGO_LEARNING_LOCAL_CACHE_PATH,
                "feature_extractors/{}".format(self.cache_identifier),
            ),
            hash=False,
            use_s3=False,
        )

    def _refresh_parameters(self, key=None):

        params = {}
        if self.pipeline_name:
            try:
                from django_learning.utils.pipelines import pipelines

                params.update(pipelines[self.pipeline_name]())
            except KeyError:
                print(
                    "WARNING: PIPELINE '{}' NOT FOUND, STICKING WITH STORED PARAMS".format(
                        self.pipeline_name
                    )
                )
                params = self.parameters
        else:
            params = self.parameters

        if key:
            self.parameters[key] = params[key]
        else:
            self.parameters = params
        self.save()
        self.refresh_from_db()  # do a handshake
        if key:
            print("Refreshed {} parameters".format(key))
        else:
            print("Refreshed all parameters")

    def _get_dataset_extractor(self, key):

        dataset_extractor = None
        try:
            from django_learning.utils.dataset_extractors import dataset_extractors

            dataset_extractor = dataset_extractors[self.parameters[key]["name"]](
                **self.parameters[key]["parameters"]
            )
        except TypeError:
            print(
                "WARNING: couldn't identify dataset extractor, it may not exist anymore!"
            )

        return dataset_extractor

    def save(self, *args, **kwargs):

        super(LearningModel, self).save(*args, **kwargs)
        self.refresh_from_db()  # pickles and unpickles self.parameters so it remains in a consistent format

    def _check_for_valid_weights(self, weight_col):
        return len(set(weight_col)) > 1 or float(weight_col.unique()[0]) != 1.0

    def extract_dataset(self, refresh=False, only_load_existing=False, **kwargs):

        test_dataset = None
        if not refresh and self.dataset_cache_hash:
            self.dataset_extractor = self._get_dataset_extractor("dataset_extractor")
            self.dataset_extractor.cache_hash = self.dataset_cache_hash
            self.dataset = self.dataset_extractor.extract(
                refresh=False, only_load_existing=only_load_existing, **kwargs
            )
            if (
                "test_dataset_extractor" in self.parameters.keys()
                and self.test_dataset_cache_hash
            ):
                test_dataset_extractor = self._get_dataset_extractor(
                    "test_dataset_extractor"
                )
                test_dataset_extractor.cache_hash = self.test_dataset_cache_hash
                test_dataset = test_dataset_extractor.extract(
                    refresh=False, only_load_existing=only_load_existing, **kwargs
                )

        if is_null(self.dataset) and not only_load_existing:

            self._refresh_parameters()  # pickles and unpickles self.parameters so it remains in a consistent format
            self.dataset_extractor = self._get_dataset_extractor(
                "dataset_extractor"
            )  # update the extractor with latest params
            self.dataset = self.dataset_extractor.extract(refresh=True, **kwargs)
            self.dataset_cache_hash = self.dataset_extractor.cache_hash
            self.save()

            if "test_dataset_extractor" in self.parameters.keys():
                test_dataset_extractor = self._get_dataset_extractor(
                    "test_dataset_extractor"
                )
                test_dataset = test_dataset_extractor.extract(refresh=True, **kwargs)
                self.test_dataset_cache_hash = test_dataset_extractor.cache_hash
                self.save()

        if is_not_null(self.dataset):
            if (
                hasattr(self.dataset_extractor, "project")
                and self.dataset_extractor.project
            ):
                self.project = self.dataset_extractor.project
                self.save()
            if not self.dataset_extractor.outcome_column:
                outcome_col = self.parameters["dataset_extractor"].get(
                    "outcome_column", None
                )
                if outcome_col:
                    self.dataset_extractor.set_outcome_column(outcome_col)
                else:
                    raise Exception(
                        "Extractor '{}' has no outcome column set and one was not specified in your pipeline".format(
                            self.parameters["dataset_extractor"]["name"]
                        )
                    )
                # self.outcome_column = self.dataset_extractor.outcome_column

        if is_not_null(test_dataset):
            test_dataset.index = test_dataset.index.map(lambda x: "test_{}".format(x))
            self.dataset = pandas.concat([self.dataset, test_dataset])

        if "training_weight" not in self.dataset.columns:
            self.dataset["training_weight"] = 1.0
        if self.parameters["model"].get("use_sample_weights", False):
            self.dataset["training_weight"] = (
                self.dataset["training_weight"] * self.dataset["sampling_weight"]
            )
        elif (
            "sampling_weight" in self.dataset.columns
            and self._check_for_valid_weights(self.dataset["sampling_weight"])
        ):
            print(
                "Okay, we'll skip the sampling weights for training, but they WILL be used in model scoring during performance evaluation"
            )

    @temp_cache_wrapper
    def load_model(
        self,
        refresh=False,
        clear_temp_cache=True,
        only_load_existing=False,
        num_cores=1,
        **kwargs
    ):

        if is_null(self.dataset):
            self.extract_dataset(only_load_existing=only_load_existing)

        cache_data = None

        if not refresh and self.model_cache_hash:
            cache_data = self.cache.read(self.model_cache_hash)
            # note: if you hit an ImportError when loading the pickle, you probably need to
            # scroll up to the top and load a utils module in this file
            # (e.g. from django_learning.utils.scoring_functions import scoring_functions)

        if is_null(cache_data) and not only_load_existing:

            # refresh the model and pipeline parameters (but not the dataset ones)
            self._refresh_parameters("model")
            self._refresh_parameters("pipeline")

            pipeline_steps = copy.copy(self.parameters["pipeline"]["steps"])
            pipeline_params = copy.copy(self.parameters["pipeline"]["params"])
            params = self._collapse_pipeline_params(pipeline_steps, pipeline_params)

            if "name" in self.parameters["model"].keys():

                from django_learning.utils.models import models as learning_models

                model_params = learning_models[self.parameters["model"]["name"]]()
                model_class = model_params.pop("model_class")
                model_params = model_params["params"]
                pipeline_steps.append(("model", model_class))

                params.update(
                    {"model__{}".format(k): v for k, v in model_params.items()}
                )
                if "params" in self.parameters["model"].keys():
                    params.update(
                        {
                            "model__{}".format(k): v
                            for k, v in self.parameters["model"]["params"].items()
                        }
                    )

            updated_hashstr = "".join(
                [
                    self.cache_identifier,
                    self.dataset_extractor.get_hash(),
                    str(get_pipeline_repr(pipeline_steps)),
                    str(get_param_repr(params)),
                    str(
                        OrderedDict(
                            sorted(
                                self.parameters.get("model", {}).items(),
                                key=lambda t: t[0],
                            )
                        )
                    ),
                ]
            )
            updated_hashstr = self.cache.file_handler.get_key_hash(updated_hashstr)
            cache_data = self._train_model(
                pipeline_steps, params, num_cores=num_cores, **kwargs
            )
            self.cache.write(updated_hashstr, cache_data, timeout=None)
            self.model_cache_hash = updated_hashstr
            self.cv_folds = None
            self.save()

        if is_not_null(cache_data):
            for k, v in cache_data.items():
                setattr(self, k, v)

    def _get_largest_code(self):

        if (
            hasattr(self.dataset_extractor, "base_class_id")
            and self.dataset_extractor.base_class_id
        ):
            largest_code = self.dataset_extractor.base_class_id
        else:
            largest_code = (
                self.dataset[self.dataset_extractor.outcome_column]
                .value_counts(ascending=False)
                .index[0]
            )

        return largest_code

    def _get_fit_params(self, train_dataset):

        fit_params = {
            "model__{}".format(k): v
            for k, v in self.parameters["model"].get("fit_params", {}).items()
        }
        # if self.parameters["model"].get("use_sample_weights", False) or self.parameters["model"].get("use_class_weights", False):
        if self._check_for_valid_weights(train_dataset["training_weight"]):
            fit_params["model__sample_weight"] = [
                x for x in train_dataset["training_weight"].values
            ]

        return fit_params

    def _train_model(self, pipeline_steps, params, num_cores=1, **kwargs):

        df = copy.copy(self.dataset)

        print("Creating train-test split")

        y = df[self.dataset_extractor.outcome_column]
        X_cols = df.columns.tolist()
        X_cols.remove(self.dataset_extractor.outcome_column)
        X = df[X_cols]
        if self.parameters["model"]["test_percent"] == 0.0:
            train_ids, test_ids = y.index, None
            print("Training on all {} cases".format(len(train_ids)))
            # if is_not_null(self.test_dataset):
            if "test_dataset_extractor" in self.parameters.keys():
                test_ids = [i for i in self.dataset.index if str(i).startswith("test_")]
                train_ids = [
                    i for i in self.dataset.index if not str(i).startswith("test_")
                ]
                # self.test_dataset['training_weight'] = 1.0
                # self.test_dataset['balancing_weight'] = 1.0
                # y_test = self.test_dataset[self.dataset_extractor.outcome_column]
                # X_test = self.test_dataset[X_cols]
                # self.test_dataset.index = self.test_dataset.index.map(lambda x: 'test_{}'.format(x))
                # test_ids = self.test_dataset.index
                # df = pandas.concat([df, self.test_dataset])
                print(
                    "Adding {} test cases from separate dataset".format(len(test_ids))
                )
        else:
            _, _, _, _, train_ids, test_ids = train_test_split(
                X,
                y,
                y.index,
                test_size=self.parameters["model"]["test_percent"],
                random_state=5,
            )
            print(
                "Selected %i training cases and %i test cases"
                % (len(train_ids), len(test_ids))
            )

        train_dataset = df.loc[train_ids]
        test_dataset = (
            df.loc[test_ids] if is_not_null(test_ids) and len(test_ids) > 0 else None
        )

        scoring_function = None
        if "scoring_function" in self.parameters["model"].keys():
            scoring_function = self._get_scoring_function(
                self.parameters["model"]["scoring_function"],
                binary_base_code=self._get_largest_code()
                if len(y.unique()) == 2
                else None,
            )

        try:
            sklearn_cache = mkdtemp(
                prefix="sklearn",
                dir=os.path.join(
                    settings.DJANGO_LEARNING_LOCAL_CACHE_PATH,
                    "feature_extractors/{}".format(self.cache_identifier),
                ),
            )
        except:
            print("Couldn't create local sklearn cache dir")
            sklearn_cache = None

        fit_params = self._get_fit_params(train_dataset)
        estimator = Pipeline(pipeline_steps, memory=sklearn_cache)

        if len(ParameterGrid(params)) == 1:
            print("Singular parameter set detected; skipping grid search")
            params = ParameterGrid(params)[0]
            model = estimator.set_params(**params)
        else:
            grid_search_cv = self.parameters["model"].get("cv", 5)

            print(
                "Beginning grid search using %s and %i cores for %s"
                % (
                    str(scoring_function),
                    num_cores,
                    self.dataset_extractor.outcome_column,
                )
            )

            model = GridSearchCV(
                estimator,
                params,
                cv=grid_search_cv,
                n_jobs=num_cores,
                verbose=2,
                scoring=scoring_function,
            )

        model.fit(
            train_dataset,
            train_dataset[self.dataset_extractor.outcome_column],
            **fit_params
        )
        if sklearn_cache:
            rmtree(sklearn_cache)

        cache_data = {
            "model": model,
            "train_dataset": train_dataset,
            "test_dataset": test_dataset,
            # "predict_dataset": predict_dataset
        }

        return cache_data

    @require_model
    def describe_model(self):

        print("'{}' results".format(self.dataset_extractor.outcome_column))

    @require_model
    def get_test_prediction_results(self, refresh=False, only_load_existing=False):

        self.predict_dataset = None
        if is_not_null(self.test_dataset):
            self.predict_dataset = self.produce_prediction_dataset(
                self.test_dataset,
                cache_key="predict_main",
                refresh=refresh,
                only_load_existing=only_load_existing,
            )
            scores = self.compute_prediction_scores(
                self.test_dataset, predicted_df=self.predict_dataset
            )
            return scores

    def print_test_prediction_report(self, refresh=False, only_load_existing=True):

        print(
            self.get_test_prediction_results(
                refresh=refresh, only_load_existing=only_load_existing
            )
        )

    @require_model
    def get_cv_prediction_results(
        self, refresh=False, only_load_existing=False, return_averages=True
    ):

        print("Computing cross-fold predictions")
        _final_model = self.model
        _final_model_best_estimator = self.model.best_estimator_
        dataset = copy.copy(self.train_dataset)

        all_fold_scores = []
        if refresh or not self.cv_folds:
            self.cv_folds = [
                f
                for f in StratifiedKFold(
                    len(dataset.index),
                    n_folds=self.parameters["model"].get("cv", 5),
                    shuffle=True,
                )
            ]
            self.save()
        for i, folds in enumerate(self.cv_folds):
            fold_train_index, fold_test_index = folds
            # NOTE: KFold returns numerical index, so you need to remap it to the dataset index (which may not be numerical)
            fold_train_dataset = dataset.ix[
                pandas.Series(dataset.index).iloc[fold_train_index].values
            ]  # self.dataset.ix[fold_train_index]
            fold_test_dataset = dataset.ix[
                pandas.Series(dataset.index).iloc[fold_test_index].values
            ]  # self.dataset.ix[fold_test_index]

            fold_predict_dataset = None
            if not refresh:
                fold_predict_dataset = self.produce_prediction_dataset(
                    fold_test_dataset,
                    cache_key="predict_fold_{}".format(i),
                    refresh=False,
                    only_load_existing=True,
                )
            if is_null(fold_predict_dataset) and not only_load_existing:
                fit_params = self._get_fit_params(fold_train_dataset)
                self.model = _final_model_best_estimator.fit(
                    fold_train_dataset,
                    fold_train_dataset[self.dataset_extractor.outcome_column],
                    **fit_params
                )
                fold_predict_dataset = self.produce_prediction_dataset(
                    fold_test_dataset,
                    cache_key="predict_fold_{}".format(i),
                    refresh=refresh,
                )

            if is_not_null(fold_predict_dataset):
                fold_scores = self.compute_prediction_scores(
                    fold_test_dataset, predicted_df=fold_predict_dataset
                )
            else:
                fold_scores = None
            all_fold_scores.append(fold_scores)

        self.model = _final_model
        if any([is_null(f) for f in all_fold_scores]):
            return None
        else:
            fold_score_df = pandas.concat(all_fold_scores)
            if return_averages:
                fold_score_df = pandas.concat(
                    [
                        all_fold_scores[0][["coder1", "coder2", "outcome_column"]],
                        fold_score_df.groupby(fold_score_df.index).mean(),
                    ],
                    axis=1,
                )

            return fold_score_df

    def print_cv_prediction_report(
        self, refresh=False, only_load_existing=True, return_averages=True
    ):

        print(
            self.get_cv_prediction_results(
                refresh=refresh,
                only_load_existing=only_load_existing,
                return_averages=return_averages,
            )
        )

    def _get_scoring_function(self, func_name, binary_base_code=None):

        try:

            from django_learning.utils.scoring_functions import scoring_functions

            scoring_function = make_scorer(scoring_functions[func_name])

        except:

            if "regression" in str(self.__class__):
                func_map = {
                    "mean_squared_error": (mean_squared_error, False, False),
                    "r2": (r2_score, True, False),
                }
                func, direction, needs_proba = func_map[func_name]
                scoring_function = make_scorer(
                    func, needs_proba=needs_proba, greater_is_better=direction
                )
            elif binary_base_code:
                func_map = {
                    "f1": (f1_score, True, False),
                    "precision": (precision_score, True, False),
                    "recall": (recall_score, True, False),
                    "brier_loss": (brier_score_loss, False, True),
                }
                func, direction, needs_proba = func_map[func_name]
                func_params = {
                    "needs_proba": needs_proba,
                    "greater_is_better": direction,
                    "pos_label": binary_base_code,
                }
                if func_name in ["f1", "precision", "recall"]:
                    func_params["labels"] = self.dataset[
                        self.dataset_extractor.outcome_column
                    ].unique()
                scoring_function = make_scorer(func, **func_params)
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
                final_params.update(
                    self._collapse_pipeline_params(step, params, names=names + [sname])
                )
        elif isinstance(pipeline, FeatureUnion):
            final_params.update(
                self._collapse_pipeline_params(
                    pipeline.transformer_list, params, names=names
                )
            )
        elif isinstance(pipeline, tuple):
            final_params.update(pipeline[1], params, names=names + [pipeline[0]])
        elif isinstance(pipeline, list):
            for sname, step in pipeline:
                final_params.update(
                    self._collapse_pipeline_params(step, params, names=names + [sname])
                )
        else:
            if names[-1] in params.keys():
                for k, v in params[names[-1]].items():
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
            from django_learning.utils.feature_extractors import BasicExtractor

            if isinstance(pipeline, BasicExtractor):
                final_params["{}__cache_identifier".format("__".join(names))] = [
                    self.cache_identifier
                ]
                final_params["{}__feature_name_prefix".format("__".join(names))] = [
                    names[-1]
                ]
                if hasattr(self, "document_types"):
                    final_params["{}__document_types".format("__".join(names))] = [
                        self.document_types
                    ]

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
    def apply_model(
        self,
        data,
        keep_cols=None,
        clear_temp_cache=True,
        disable_probability_threshold_warning=False,
    ):

        if not keep_cols:
            keep_cols = []

        predictions = self.model.predict(data)
        has_probabilities = False
        try:
            probabilities = self.model.predict_proba(data)
            has_probabilities = True
        except AttributeError:
            probabilities = [None] * len(data)

        labels = []
        for index, pred, prob in zip(data.index, predictions, probabilities):
            if type(prob) in [list, tuple, numpy.ndarray]:
                prob = max(prob)
            label = {self.dataset_extractor.outcome_column: pred, "probability": prob}
            for col in keep_cols:
                label[col] = data.loc[index, col]
            labels.append(label)

        labels = pandas.DataFrame(labels, index=data.index)
        if not has_probabilities:
            del labels["probability"]

        return labels

    @require_model
    def produce_prediction_dataset(
        self,
        df_to_predict,
        cache_key=None,
        refresh=False,
        only_load_existing=False,
        disable_probability_threshold_warning=False,
    ):

        from django_learning.utils.dataset_extractors import dataset_extractors

        predicted_df = dataset_extractors["model_prediction_dataset"](
            dataset=df_to_predict,
            learning_model=self,
            cache_key=cache_key,
            disable_probability_threshold_warning=disable_probability_threshold_warning,
        ).extract(refresh=refresh, only_load_existing=only_load_existing)

        return predicted_df

    @require_model
    def compute_prediction_scores(
        self,
        df_to_predict,
        predicted_df=None,
        cache_key=None,
        refresh=False,
        only_load_existing=False,
    ):

        if (
            "sampling_weight" in df_to_predict.columns
            and self._check_for_valid_weights(df_to_predict["sampling_weight"])
        ):
            weight_col = "sampling_weight"
        else:
            weight_col = None
        if is_null(predicted_df):
            predicted_df = self.produce_prediction_dataset(
                df_to_predict,
                cache_key=cache_key,
                refresh=refresh,
                only_load_existing=only_load_existing,
            )
        if is_not_null(predicted_df):
            return compute_scores_from_datasets_as_coders(
                df_to_predict,
                predicted_df,
                "index",
                self.dataset_extractor.outcome_column,
                weight_column=weight_col,
            )
        else:
            return None


class DocumentLearningModel(LearningModel):

    sampling_frame = models.ForeignKey(
        "django_learning.SamplingFrame",
        related_name="learning_models",
        on_delete=models.CASCADE,
    )
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

    def save(self, *args, **kwargs):

        super(DocumentLearningModel, self).save(*args, **kwargs)

    def extract_dataset(self, refresh=False, **kwargs):

        super(DocumentLearningModel, self).extract_dataset(refresh=refresh, **kwargs)
        if (
            hasattr(self.dataset_extractor, "sampling_frame")
            and self.dataset_extractor.sampling_frame
        ):
            self.sampling_frame = self.dataset_extractor.sampling_frame
            self.save()
