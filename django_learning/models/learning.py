# import importlib, copy, pandas
#
# from django.db import models
# from django.contrib.postgres.fields import ArrayField
# from django.conf import settings
#
# from picklefield.fields import PickledObjectField
# from langdetect import detect
# from abc import abstractmethod
# from collections import OrderedDict
#
# from sklearn.cross_validation import train_test_split
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss, make_scorer, mean_squared_error, r2_score
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.metrics import classification_report, confusion_matrix
#
# from django_commander.models import LoggedExtendedModel
#
# from django_learning.utils import get_document_types, pipelines, training_data_extractors, get_pipeline_repr, get_param_repr
# from django_learning.utils.decorators import require_training_data, require_model, temp_cache_wrapper
# from django_learning.utils.feature_extractors import BasicExtractor
# from django_learning.utils import models as learning_models
#
# from pewtils import is_not_null, is_null, decode_text, recursive_update
# from pewtils.django import get_model, CacheHandler
#
#
#
# class Classification(LoggedExtendedModel):
#
# 	document = models.ForeignKey("django_learning.Document", related_name="classifications")
# 	label = models.ForeignKey("django_learning.Label", related_name="classifications")
#
# 	classifier = models.ForeignKey("django_learning.Classifier", related_name="classifications")
# 	probability = models.FloatField(null=True, help_text="The probability of the assigned label, if applicable")
#
#     # def validate_unique(self, *args, **kwargs):
#     #        super(ClassifierDocumentCode, self).validate_unique(*args, **kwargs)
#     #        if not self.id:
#     #            if self.__class__.objects.filter(code__variable=self.code.variable).exists():
#     #                raise ValidationError(
#     #                    {
#     #                        NON_FIELD_ERRORS: [
#     #                            'ClassifierDocumentCode with the same variable already exists'
#     #                        ],
#     #                    }
#     #                )
#
#     #    def __repr__(self):
#     #        return "<ClassifierDocumentCode value={0}, code__variable={1}, document_id={2}>".format(
#     #            self.code.label, self.code.variable.name, self.document.id
#     #        )
#
#
#
# class LearningModel(LoggedExtendedModel):
#
#     name = models.CharField(max_length=100, unique=True, help_text="Unique name of the classifier")
#     outcome_column = models.CharField(max_length=256)
#     pipeline_name = models.CharField(max_length=150, null=True,
#                                      help_text="The named pipeline used to seed the handler's parameters, if any; note that the JSON pipeline file may have changed since this classifier was created; refer to the parameters field to view the exact parameters used to compute the model")
#     parameters = PickledObjectField(null=True,
#                                     help_text="A pickle file of the parameters used to process the codes and generate the model")
#
#     cv_folds = PickledObjectField(null=True)
#     cv_folds_test = PickledObjectField(null=True)
#
#     training_data_hash = models.CharField(max_length=256, null=True)
#     model_hash =  models.CharField(max_length=256, null=True)
#
#     class Meta:
#
#         abstract = True
#
#     def __init__(self, *args, **kwargs):
#
#         self.cache_identifier = "{}_{}".format(self.name, self.outcome_column)
#
#         self.model = None
#         self.X = None
#         self.Y = None
#         self.train_x = None
#         self.train_y = None
#         self.train_ids = None
#         self.test_x = None
#         self.test_y = None
#         self.test_ids = None
#         self.predict_y = None
#
#         params = {}
#         if self.pipeline_name:
#             params.update(pipelines[self.pipeline_name]())
#         self.parameters = recursive_update(params, self.parameters if self.parameters else {})
#
#         self.cache = CacheHandler("learning/supervised/{}".format(self.cache_identifier),
#             use_s3=True,
#             aws_access=settings.DJANGO_LEARNING_AWS_ACCESS,
#             aws_secret=settings.DJANGO_LEARNING_AWS_SECRET
#         )
#         self.temp_cache = CacheHandler("learning/feature_extractors/{}".format(self.cache_identifier), use_s3=False)
#
#         super(LearningModel, self).__init__(*args, **kwargs)
#
#     def save(self, *args, **kwargs):
#
#         self.parameters['pipeline']['steps'] = [(k, v) for k, v in self.parameters['pipeline']['steps'] if k != 'model']
#         # TODO: figure out where this bug is occuring, but for now we'll deal with extra "model" params sneaking in
#
#         super(LearningModel, self).save(*args, **kwargs)
#
#     def _get_training_data_hash(self):
#
#         return "".join([
#             self.cache_identifier,
#             self.parameters.get("training_data_extractor", ""),
#             str(OrderedDict(sorted(self.parameters.get("codes", {}).items(), key=lambda t: t[0]))),
#             str(OrderedDict(sorted(self.parameters.get("documents", {}).items(), key=lambda t: t[0])))
#         ])
#
#     def _get_training_data(self):
#
#         return training_data_extractors[self.parameters["training_data_extractor"]]()
#
#     def load_training_data(self, refresh=False, only_load_existing=False, **kwargs):
#
#         cache_data = None
#
#         if not refresh:
#             if not self.training_data_hash:
#                 self.training_data_hash = self._get_training_data_hash()
#             cache_data = self.cache.read(self.training_data_hash)
#
#         if is_null(cache_data) and not only_load_existing:
#
#             updated_hashstr = self._get_training_data_hash()
#             cache_data = self._get_training_data()
#             self.cache.write(updated_hashstr, cache_data)
#             self.training_data_hash = updated_hashstr
#             self.save()
#
#         if is_not_null(cache_data):
#             for k, v in cache_data.iteritems():
#                 setattr(self, k, v)
#
#     @temp_cache_wrapper
#     def load_model(self, refresh=False, clear_temp_cache=True, only_load_existing=False, **kwargs):
#
#         cache_data = None
#
#         if not refresh and self.model_hash:
#             cache_data = self.cache.read(self.model_hash)
#
#         if is_null(cache_data) and not only_load_existing:
#
#             pipeline_steps = copy.copy(self.parameters['pipeline']['steps'])
#             params = self._collapse_pipeline_params(
#                 pipeline_steps,
#                 self.parameters['pipeline']['params']
#             )
#
#             if "name" in self.parameters["model"].keys():
#
#                 model_params = learning_models[self.parameters["model"]["name"]]()
#                 model_class = model_params.pop("model_class")
#                 model_params = model_params["params"]
#                 pipeline_steps.append(("model", model_class))
#
#                 params.update({"model__{}".format(k): v for k, v in model_params.iteritems()})
#                 if 'params' in self.parameters['model'].keys():
#                     params.update({"model__{}".format(k): v for k, v in self.parameters['model']['params'].iteritems()})
#
#             updated_hashstr = "".join([
#                 self.cache_identifier,
#                 self.parameters.get("training_data_extractor", ""),
#                 str(OrderedDict(sorted(self.parameters.get("codes", {}).items(), key=lambda t: t[0]))),
#                 str(OrderedDict(sorted(self.parameters.get("documents", {}).items(), key=lambda t: t[0]))),
#                 str(get_pipeline_repr(pipeline_steps)),
#                 str(get_param_repr(params)),
#                 str(OrderedDict(sorted(self.parameters.get("model", {}).items(), key=lambda t: t[0])))
#             ])
#             cache_data = self._train_model(pipeline_steps, params, **kwargs)
#             self.cache.write(updated_hashstr, cache_data)
#             self.model_hash = updated_hashstr
#             self.save()
#
#         if is_not_null(cache_data):
#             for k, v in cache_data.iteritems():
#                 setattr(self, k, v)
#
#     @require_training_data
#     def _train_model(self, pipeline_steps, params, **kwargs):
#
#         df = self.training_data
#
#         smallest_code = df[self.outcome_column].value_counts(ascending=True).index[0]
#         largest_code = df[self.outcome_column].value_counts(ascending=False).index[0]
#
#         # print "Code frequencies: {}".format(dict(df[self.outcome_variable].value_counts(ascending=True)))
#
#         if "training_weight" not in df.columns:
#             df["training_weight"] = 1.0
#
#         if self.parameters["model"].get("use_class_weights", False):
#
#             class_weights = {}
#             base_weight = df[df[self.outcome_column] == largest_code]['training_weight'].sum()
#             # total_weight = df['training_weight'].sum()
#             for c in df[self.outcome_variable].unique():
#                 # class_weights[c] = float(df[df[self.outcome_column]==c]["training_weight"].sum()) / float(total_weight)
#                 class_weights[c] = base_weight / float(df[df[self.outcome_column] == c]['training_weight'].sum())
#             total_weight = sum(class_weights.values())
#             class_weights = {k: float(v) / float(total_weight) for k, v in class_weights.items()}
#             params["model__class_weight"] = [class_weights, ]
#             print "Class weights: {}".format(class_weights)
#
#         print "Creating train-test split"
#
#         y = df[self.outcome_column]
#         X_cols = df.columns.tolist()
#         X_cols.remove(self.outcome_column)
#         X = df[X_cols]
#         if self.parameters["model"]["test_percent"] == 0.0:
#             X_train, X_test, y_train, y_test, train_ids, test_ids = X, None, y, None, y.index, None
#             print "Training on all {} cases".format(len(y_train))
#         else:
#             X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(X, y, y.index, test_size=
#             self.parameters["model"]["test_percent"], random_state=5)
#             print "Selected %i training cases and %i test cases" % (
#                 len(y_train),
#                 len(y_test)
#             )
#
#         scoring_function = None
#         if "scoring_function" in self.parameters["model"].keys():
#             scoring_function = self._get_scoring_function(
#                 self.parameters["model"]["scoring_function"],
#                 binary_base_code=smallest_code if len(y.unique()) == 2 else None
#             )
#
#         print "Beginning grid search using %s and %i cores for %s" % (
#             str(scoring_function),
#             self.num_cores,
#             self.outcome_column
#         )
#
#         model = GridSearchCV(
#             Pipeline(pipeline_steps),
#             params,
#             fit_params={'model__sample_weight': [x for x in X_train["training_weight"].values]} if self.parameters[
#                 "model"].get("use_sample_weights", False) else {},
#             cv=self.parameters["model"].get("cv", 5),
#             n_jobs=self.num_cores,
#             verbose=1,
#             scoring=scoring_function
#         )
#
#         model.fit(X_train, y_train)
#
#         print "Finished training model, best score: {}".format(model.best_score_)
#
#         predict_y = model.predict(X_test) if is_not_null(X_test) else None
#
#         cache_data = {
#             "model": model,
#             "train_x": X_train,
#             "train_y": y_train,
#             "train_ids": train_ids,
#             "test_x": X_test,
#             "test_y": y_test,
#             "test_ids": test_ids,
#             "predict_y": predict_y
#         }
#
#         return cache_data
#
#     def _get_scoring_function(self, func_name, binary_base_code=None):
#
#         try:
#
#             from django_learning.utils.scoring_functions import scoring_functions
#             scoring_function = make_scorer(scoring_functions[func_name].scorer)
#
#         except:
#
#             if "regression" in str(self.__class__):
#                 func_map = {
#                     "mean_squared_error": (mean_squared_error, False, False),
#                     "r2": (r2_score, True, False)
#                 }
#                 func, direction, needs_proba = func_map[func_name]
#                 scoring_function = make_scorer(func, needs_proba=needs_proba, greater_is_better=direction)
#             elif binary_base_code:
#                 func_map = {
#                     "f1": (f1_score, True, False),
#                     "precision": (precision_score, True, False),
#                     "recall": (recall_score, True, False),
#                     "brier_loss": (brier_score_loss, False, True)
#                 }
#                 func, direction, needs_proba = func_map[func_name]
#                 scoring_function = make_scorer(func, needs_proba=needs_proba, greater_is_better=direction,
#                                                pos_label=binary_base_code)
#             else:
#                 if self.parameters["model"]["scoring_function"] == "f1":
#                     scoring_function = "f1_macro"
#                     # scoring_function = "f1_micro"
#                     # scoring_function = "f1_weighted"
#                 elif self.parameters["model"]["scoring_function"] == "precision":
#                     scoring_function = "precision"
#                 else:
#                     scoring_function = "recall"
#
#         return scoring_function
#
#     def _collapse_pipeline_params(self, pipeline, params, names=None):
#
#         final_params = {}
#         if not names:
#             names = []
#         if isinstance(pipeline, Pipeline):
#             for sname, step in pipeline.steps:
#                 final_params.update(self._collapse_pipeline_params(step, params, names=names + [sname]))
#         elif isinstance(pipeline, FeatureUnion):
#             final_params.update(self._collapse_pipeline_params(pipeline.transformer_list, params, names=names))
#         elif isinstance(pipeline, tuple):
#             final_params.update(pipeline[1], params, names=names + [pipeline[0]])
#         elif isinstance(pipeline, list):
#             for sname, step in pipeline:
#                 final_params.update(self._collapse_pipeline_params(step, params, names=names + [sname]))
#         else:
#             if names[-1] in params.keys():
#                 for k, v in params[names[-1]].iteritems():
#                     # if k == "preprocessors":
#                     #     preprocessor_sets = []
#                     #     for pset in v:
#                     #         preprocessors = []
#                     #         try:
#                     #             for preprocessor_name, preprocessor_params in pset:
#                     #                 preprocessor_module = importlib.import_module("logos.learning.utils.preprocessors.{0}".format(preprocessor_name))
#                     #                 preprocessors.append(preprocessor_module.Preprocessor(**preprocessor_params))
#                     #         except ValueError: pass
#                     #         preprocessor_sets.append(preprocessors)
#                     #     v = preprocessor_sets
#                     if len(v) > 0:
#                         final_params["__".join(names + [k])] = v
#             if isinstance(pipeline, BasicExtractor):
#                 final_params["{}__cache_identifier".format("__".join(names))] = [self.cache_identifier]
#                 final_params["{}__feature_name_prefix".format("__".join(names))] = [names[-1]]
#                 if hasattr(self, 'document_types'):
#                     final_params["{}__document_types".format("__".join(names))] = [self.document_types]
#
#         return final_params
#
#     @require_model
#     def get_feature_names(self, m):
#
#         features = []
#
#         if hasattr(m, "steps"):
#             for name, step in m.steps:
#                 features.append(self.get_feature_names(step))
#         elif hasattr(m, "transformer_list"):
#             for name, step in m.transformer_list:
#                 features.append(self.get_feature_names(step))
#         elif hasattr(m, "get_feature_names"):
#             return m.get_feature_names()
#
#         return [f for sublist in features for f in sublist]
#
#     @require_model
#     def print_report(self):
#
#         print "'%s' results" % self.outcome_variable
#
#         print "Best score: {} ({} std.)".format(self.model.best_score_,
#                                                 getattr(self.model, "best_score_std_", None))
#
#         # print "Best parameters:"
#         # params = self.model.best_params_
#         # for p in params.keys():
#         #     if p.endswith("__stop_words"):
#         #         del params[p]
#         # print params
#
#     @require_model
#     @temp_cache_wrapper
#     def apply_model(self, data, keep_cols=None, clear_temp_cache=True):
#
#         pass
#         if not keep_cols: keep_cols = []
#
#         predictions = self.model.predict(data)
#         try:
#             probabilities = self.model.predict_proba(data)
#         except AttributeError:
#             probabilities = [None] * len(data)
#
#         codes = []
#         for index, pred, prob in zip(data.index, predictions, probabilities):
#             if type(prob) == list or type(prob) == tuple:
#                 prob = max(prob)
#             code = {
#                 self.outcome_column: pred,
#                 "probability": prob
#             }
#             for col in keep_cols:
#                 code[col] = data.loc[index, col]
#             codes.append(code)
#
#         return pandas.DataFrame(codes)
#
#
# class ClassificationLearningModel(LearningModel):
#
#     @require_model
#     def show_top_features(self, n=10):
#
#         if hasattr(self.model.best_estimator_, "named_steps"):
#             steps = self.model.best_estimator_.named_steps
#         else:
#             steps = self.model.best_estimator_.steps
#
#         feature_names = self.get_feature_names(self.model.best_estimator_)
#         class_labels = steps['model'].classes_
#
#         top_features = {}
#         if hasattr(steps['model'], "coef_"):
#             if len(class_labels) == 2:
#                 top_features[0] = sorted(zip(
#                     steps['model'].coef_[0],
#                     feature_names
#                 ))[:n]
#                 top_features[1] = sorted(zip(
#                     steps['model'].coef_[0],
#                     feature_names
#                 ))[:-(n + 1):-1]
#             else:
#                 for i, class_label in enumerate(class_labels):
#                     top_features[class_label] = sorted(zip(
#                         steps['model'].coef_[i],
#                         feature_names
#                     ))[-n:]
#         elif hasattr(steps['model'], "feature_importances_"):
#             top_features["n/a"] = sorted(zip(
#                 steps['model'].feature_importances_,
#                 feature_names
#             ))[:-(n + 1):-1]
#
#         for class_label, top_n in top_features.iteritems():
#             print class_label
#             for c, f in top_n:
#                 try:
#                     print "\t%.4f\t\t%-15s" % (c, f)
#                 except:
#                     print "Error: {}, {}".format(c, f)
#
#     @require_model
#     def print_report(self):
#
#         super(ClassificationLearningModel, self).print_report()
#
#         print "Detailed classification report:"
#         print classification_report(self.test_y, self.predict_y,
#                                     sample_weight=self.test_x['sampling_weight'] if self.parameters["model"].get(
#                                         "use_sample_weights", False) else None)
#
#         print "Confusion matrix:"
#         print confusion_matrix(self.test_y, self.predict_y)
#
#         self.show_top_features()
#
#     @require_model
#     def get_incorrect_predictions(self, actual_code=None):
#
#         df = pandas.concat([self.test_y, self.test_x], axis=1)
#         df['prediction'] = self.predict_y
#         df = df[df[self.outcome_column] != df['prediction']]
#         if actual_code:
#             df = df[df[self.outcome_column] == actual_code]
#         return df
#
#     @require_model
#     def get_report_results(self):
#
#         rows = []
#         report = classification_report(self.test_y, self.predict_y,
#                                        sample_weight=self.test_x['sampling_weight'] if self.parameters["model"].get(
#                                            "use_sample_weights", False) else None)
#         for row in report.split("\n"):
#             row = row.strip().split()
#             if len(row) == 7:
#                 row = row[2:]
#             if len(row) == 5:
#                 rows.append({
#                     "class": row[0],
#                     "precision": row[1],
#                     "recall": row[2],
#                     "f1-score": row[3],
#                     "support": row[4]
#                 })
#         return rows
#
# class RegressionLearningModel(LearningModel):
#
#     @require_model
#     def show_top_features(self, n=10):
#
#         feature_names = self.get_feature_names(self.model.best_estimator_)
#
#         if hasattr(self.model.best_estimator_.named_steps['model'], "coef_"):
#             top_features = sorted(zip(
#                 self.model.best_estimator_.named_steps['model'].coef_,
#                 feature_names
#             ))[:-(n + 1):-1]
#         elif hasattr(self.model.best_estimator_.named_steps['model'], "feature_importances_"):
#             top_features = sorted(zip(
#                 self.model.best_estimator_.named_steps['model'].feature_importances_,
#                 feature_names
#             ))[:-(n + 1):-1]
#         for c, f in top_features:
#             print "\t%.4f\t\t%-15s" % (c, f)
#
#     @require_model
#     def print_report(self):
#
#         super(RegressionLearningModel, self).print_report()
#
#         self.show_top_features()
#
#
# class DocumentLearningModel(LearningModel):
#
#     frame = models.ForeignKey("django_learning.DocumentSampleFrame", related_name="learning_models")
#     question = models.ForeignKey("django_learning.Question", related_name="learning_models")
#
#     class Meta:
#
#         abstract = True
#
#     def __init__(self, *args, **kwargs):
#
#         self.outcome_column = "label_id"
#
#         super(DocumentLearningModel, self).__init__(*args, **kwargs)
#
#
# class DocumentClassificationLearningModel(ClassificationLearningModel, DocumentLearningModel):
#
#     def __init__(self, *args, **kwargs):
#
#         super(DocumentClassificationLearningModel, self).__init__(*args, **kwargs)
#
#         # Always pre-compute unnecessary/bad code parameters and delete them, to keep cache keys consistent:
#
#         def has_all_params(p):
#             return all([k in p.keys() for k in [
#                 "code_filters",
#                 "consolidation_threshold",
#                 "consolidation_min_quantile"
#             ]])
#
#         has_experts = self._has_raw_codes(turk=False)
#         self.use_expert_codes = False
#         if "experts" in self.parameters["codes"].keys():
#             if is_not_null(self.parameters["codes"]["experts"]) and has_all_params(self.parameters["codes"]["experts"]):
#                 if has_experts:
#                     self.use_expert_codes = True
#                 else:
#                     del self.parameters["codes"]["experts"]
#             else:
#                 del self.parameters["codes"]["experts"]
#
#         has_mturk = self._has_raw_codes(turk=True)
#         self.use_mturk_codes = False
#         if "mturk" in self.parameters["codes"].keys():
#             if is_not_null(self.parameters["codes"]["mturk"]) and has_all_params(self.parameters["codes"]["mturk"]):
#                 if has_mturk:
#                     self.use_mturk_codes = True
#                 else:
#                     del self.parameters["codes"]["mturk"]
#             else:
#                 del self.parameters["codes"]["mturk"]
#
#         fallback = False
#         if self.use_expert_codes and self.use_mturk_codes:
#             if not has_all_params(self.parameters["codes"]):
#                 fallback = True
#                 for k in ["code_filters", "consolidation_threshold", "consolidation_min_quantile",
#                           "mturk_to_expert_weight"]:
#                     if k in self.parameters["codes"]:
#                         del self.parameters["codes"][k]
#         else:
#             for k in ["code_filters", "consolidation_threshold", "consolidation_min_quantile",
#                       "mturk_to_expert_weight"]:
#                 if k in self.parameters["codes"]:
#                     del self.parameters["codes"][k]
#
#         if fallback:
#             if has_experts and self.use_expert_codes:
#                 self.use_mturk_codes = False
#                 if "mturk" in self.parameters["codes"]:
#                     del self.parameters["codes"]["mturk"]
#             elif has_mturk and self.use_mturk_codes:
#                 self.use_expert_codes = False
#                 if "experts" in self.parameters["codes"]:
#                     del self.parameters["codes"]["experts"]
#             else:
#                 self.use_mturk_codes = False
#                 self.use_expert_codes = False
#
#     def _get_training_data(self, validation=False, **kwargs):
#
#         from django_learning.utils.code_filters import code_filters
#
#         expert_codes = None
#         if self.use_expert_codes:
#
#             print "Extracting expert codes"
#             expert_codes = self._get_raw_codes(turk=False, training=(not validation))
#             print "{} expert codes extracted".format(len(expert_codes))
#
#             for filter_name, filter_params in self.parameters["codes"]["experts"]["code_filters"]:
#                 if is_not_null(expert_codes, empty_lists_are_null=True):
#                     print "Applying expert code filter: %s" % filter_name
#                     expert_codes = code_filters[filter_name](expert_codes, **filter_params)
#                     # filter_module = importlib.import_module("logos.learning.utils.code_filters.{0}".format(filter_name))
#                     # expert_codes = filter_module.filter(expert_codes, **filter_params)
#
#             if is_not_null(expert_codes):
#
#                 if self.parameters["codes"]["experts"]["consolidation_threshold"]:
#                     print "Consolidating expert codes at threshold {}".format(
#                         self.parameters["codes"]["experts"]["consolidation_threshold"])
#                     expert_codes = self._consolidate_codes(
#                         expert_codes,
#                         threshold=self.parameters["codes"]["experts"]["consolidation_threshold"],
#                         keep_quantile=self.parameters["codes"]["experts"]["consolidation_min_quantile"],
#                         fake_coder_id="experts"
#                     )
#
#         mturk_codes = None
#         if self.use_mturk_codes:
#
#             print "Extracting MTurk codes"
#             mturk_codes = self._get_raw_codes(turk=True, training=(not validation))
#             print "{} MTurk codes extracted".format(len(mturk_codes))
#
#             if is_not_null(mturk_codes, empty_lists_are_null=True):
#
#                 for filter_name, filter_params in self.parameters["codes"]["mturk"]["code_filters"]:
#                     print "Applying MTurk code filter: %s" % filter_name
#                     mturk_codes = code_filters[filter_name](mturk_codes, **filter_params)
#                     # filter_module = importlib.import_module("logos.learning.utils.code_filters.{0}".format(filter_name))
#                     # mturk_codes = filter_module.filter(mturk_codes, **filter_params)
#
#                 if is_not_null(mturk_codes):
#
#                     if self.parameters["codes"]["mturk"]["consolidation_threshold"]:
#                         print "Consolidating Mturk codes at threshold {}".format(
#                             self.parameters["codes"]["mturk"]["consolidation_threshold"])
#                         mturk_codes = self._consolidate_codes(
#                             mturk_codes,
#                             threshold=self.parameters["codes"]["mturk"]["consolidation_threshold"],
#                             keep_quantile=self.parameters["codes"]["mturk"]["consolidation_min_quantile"],
#                             fake_coder_id="mturk"
#                         )
#
#         if self.use_expert_codes and self.use_mturk_codes:
#
#             codes = pandas.concat([c for c in [expert_codes, mturk_codes] if is_not_null(c)])
#
#             for filter_name, filter_params in self.parameters["codes"]["code_filters"]:
#                 print "Applying global code filter: %s" % filter_name
#                 codes = code_filters[filter_name](codes, **filter_params)
#                 # filter_module = importlib.import_module("logos.learning.utils.code_filters.{0}".format(filter_name))
#                 # codes = filter_module.filter(codes, **filter_params)
#
#             print "Consolidating all codes"
#
#             if "mturk_to_expert_weight" in self.parameters["codes"]:
#                 df = self._consolidate_codes(
#                     codes,
#                     threshold=self.parameters["codes"]["consolidation_threshold"],
#                     keep_quantile=self.parameters["codes"]["consolidation_min_quantile"],
#                     mturk_to_expert_weight=self.parameters["codes"]["mturk_to_expert_weight"]
#                 )
#             else:
#                 df = self._consolidate_codes(
#                     codes,
#                     threshold=self.parameters["codes"]["consolidation_threshold"],
#                     keep_quantile=self.parameters["codes"]["consolidation_min_quantile"]
#                 )
#
#         elif self.use_expert_codes:
#             df = expert_codes
#         elif self.use_mturk_codes:
#             df = mturk_codes
#         else:
#             df = pandas.DataFrame()
#
#         df = df[df['code_id'].notnull()]
#         df = self._add_code_labels(df)
#
#         input_documents = self.filter_documents(
#             get_model("Document").objects.filter(pk__in=df["document_id"])
#         )
#
#         training_data = df.merge(
#             input_documents,
#             how='inner',
#             left_on='document_id',
#             right_on="pk"
#         )
#
#         if self.parameters["documents"].get("include_frame_weights", False):
#             training_data = self._add_frame_weights(training_data)
#
#         training_data = self._add_balancing_weights(training_data)
#
#         return {
#             "training_data": training_data
#         }
#
#
# # class DocumentClassificationModel(DocumentLearningModel):
# #
# #     """
# #     Holds data for a classifier for a given variable and type of document.  Only one classifier may exist for a given
# #     combination of code variable and document type - to keep things simple and uncomplicated (we don't want competing
# #     alternative versions of a model that do the same thing.)  Holds pickled objects that contain the training/test data
# #     and parameters used to train the model, and the actual trained model itself.
# #     """
# #
# #     variable = models.ForeignKey("django_learning.CodeVariable", related_name="classifiers",
# #                                  help_text="The code variable whose codes the classifier is attempting to assign")
# #
# #     objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()
# #
# #     def compute_cv_folds(self, use_test_data=False, num_folds=5, refresh=False):
# #
# #         h = self.handler
# #         fold_preds = None
# #         if (not use_test_data and (is_null(self.cv_folds) or refresh)) or \
# #                 (use_test_data and (is_null(self.cv_folds_test) or refresh)):
# #             fold_preds = h.compute_cv_folds(use_test_data=use_test_data, num_folds=num_folds)
# #         if fold_preds:
# #             if use_test_data:
# #                 self.cv_folds_test = fold_preds
# #             else:
# #                 self.cv_folds = fold_preds
# #             self.save()
# #
# #     def get_code_cv_training_scores(self, use_test_data=False, code_value="1", partition_by=None,
# #                                     restrict_document_type=None, min_support=0):
# #
# #         h = self.handler
# #         h.load_model(only_load_existing=True)
# #         if is_not_null(h.model):
# #
# #             if use_test_data:
# #                 X = h.test_x
# #                 y = h.test_y
# #                 fold_preds = self.cv_folds_test
# #             else:
# #                 X = h.train_x
# #                 y = h.train_y
# #                 fold_preds = self.cv_folds
# #
# #             return h.get_code_cv_training_scores(
# #                 fold_preds, X, y,
# #                 code_value=code_value,
# #                 partition_by=partition_by,
# #                 restrict_document_type=restrict_document_type,
# #                 min_support=min_support
# #             )
# #
# #         else:
# #             print "Couldn't find cached model to load using the saved parameters"
# #             return None
# #
# #     def get_code_validation_test_scores(self, code_value="1", partition_by=None, restrict_document_type=None,
# #                                         use_expert_consensus_subset=False, compute_for_experts=False, min_support=0):
# #
# #         h = self.handler
# #         h.load_model(only_load_existing=True)
# #         if is_not_null(h.model):
# #
# #             return h.get_code_validation_test_scores(
# #                 code_value=code_value,
# #                 partition_by=partition_by,
# #                 restrict_document_type=restrict_document_type,
# #                 use_expert_consensus_subset=use_expert_consensus_subset,
# #                 compute_for_experts=compute_for_experts,
# #                 min_support=min_support
# #             )
# #
# #         else:
# #             print "Couldn't find cached model to load using the saved parameters"
# #             return None
# #
# #     def show_top_features(self, n=10):
# #
# #         self.handler.show_top_features(n=n)
# #
# #     def print_report(self):
# #
# #         self.handler.print_report()
# #
# #     def get_report_results(self):
# #
# #         return self.handler.get_report_results()
# #
# #     def apply_model_to_frames(self, num_cores=2, chunk_size=1000, refresh_existing=False):
# #
# #         h = self.handler
# #         h.load_model(only_load_existing=True)
# #         if is_not_null(h.model):
# #             print "Selecting frame documents"
# #             docs = get_model("Document").objects.filter(sample_frames__in=self.frames.all())
# #             if not refresh_existing:
# #                 existing = self.coded_documents.values_list("document_id", flat=True)
# #                 keep = get_model("Document").objects.filter(
# #                     pk__in=set(docs.values_list("pk", flat=True)).difference(set(existing)))
# #                 print "Skipping {} existing documents, {} remaining".format(existing.count(), keep.count())
# #                 # if existing.count() > 0:
# #                 #    docs = docs.exclude(pk__in=existing)
# #                 docs = keep
# #             print "Applying model to {} documents".format(docs.count())
# #             h.apply_model_to_database(docs, chunk_size=chunk_size, num_cores=num_cores)
# #
# #
# # class DocumentRegressionModel(DocumentLearningModel):
# #
# #     pass
