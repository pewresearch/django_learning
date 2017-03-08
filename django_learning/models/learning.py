# import importlib
#
#
# from django.db import models
# from django.contrib.postgres.fields import ArrayField
#
# from picklefield.fields import PickledObjectField
# from langdetect import detect
# from abc import abstractmethod
# from collections import OrderedDict
#
# from django_learning.settings import DJANGO_LEARNING_BASE_MODEL, DJANGO_LEARNING_BASE_MANAGER
# from django_learning.utils import get_document_types
#
# from pewtils import is_not_null, is_null, decode_text
# from pewtils.django import get_model
#
#
#
# class Classification(models.Model):
#
# 	document = models.ForeignKey("django_learning.Document", related_name="classifications")
# 	label = models.ForeignKey("django_learning.Label", related_name="classifications")
#
# 	classifier = models.ForeignKey("django_learning.Classifier", related_name="classifications")
# 	probability = models.FloatField(null=True, help_text="The probability of the assigned label, if applicable")
#
#     objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()
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
# class LearningModel(DJANGO_LEARNING_BASE_MODEL):
#
#     name = models.CharField(max_length=100, unique=True, help_text="Unique name of the classifier")
#     handler_class = models.CharField(max_length=100, default="DocumentClassificationHandler")
#     pipeline_name = models.CharField(max_length=150, null=True,
#                                      help_text="The named pipeline used to seed the handler's parameters, if any; note that the JSON pipeline file may have changed since this classifier was created; refer to the parameters field to view the exact parameters used to compute the model")
#     parameters = PickledObjectField(null=True,
#                                     help_text="A pickle file of the parameters used to process the codes and generate the model")
#     cv_folds = PickledObjectField(null=True)
#     cv_folds_test = PickledObjectField(null=True)
#
#
#     training_data_hash = models.CharField(max_length=256, null=True)
#     model_hash =  models.CharField(max_length=256, null=True)
#     # only update the hashes when you save the training_data or model
#     # that way, even if the parameters change, you can always recover the S3 files
#     # they just may get renamed if the configs change, but this thing should load using the saved hashes
#     # and save using updated/recomputed hashes (which then get saved to point to the new files)
#
#     objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()
#
#     class Meta:
#
#         abstract = True
#
#     def __init__(self, *args, **kwargs):
#
#         self.model = None
#         self.train_x = None
#         self.train_y = None
#         self.train_ids = None
#         self.test_x = None
#         self.test_y = None
#         self.test_ids = None
#         self.predict_y = None
#
#         # if not parameters, then load the file based on the pipeline name
#         super(LearningModel, self).__init__(*args, **kwargs)
#
#     @property
#     def training_data(self):
#         pass
#
#     def load_training_data(self, refresh=False, only_load_existing=False, **kwargs):
#
#         cache_data = None
#
#         if not refresh:
#             cache_data = self.cache.read(self.training_data_hash)
#
#         if is_null(cache_data) and not only_load_existing:
#
#             updated_hashstr = "".join([
#                 self.cache_identifier,
#                 str(OrderedDict(sorted(self.parameters.get("codes", {}).items(), key=lambda t: t[0]))),
#                 str(OrderedDict(sorted(self.parameters.get("documents", {}).items(), key=lambda t: t[0])))
#             ])
#
#             cache_data = self._get_training_data(**kwargs)
#             self.cache.write(updated_hashstr, cache_data)
#             self.training_data_hash = updated_hashstr
#             self.save()
#
#         if is_not_null(cache_data):
#             for k, v in cache_data.iteritems():
#                 setattr(self, k, v)
#
#     @abstractmethod
#     def _get_training_data(self):
#         raise NotImplementedError
#
#     def load_model(selfs, refresh=False, clear_temp_cache=True, only_load_existing=False, **kwargs):
#
#         pass
#         # pipeline_steps = copy.copy(self.parameters['pipeline']['steps'])
#         # params = self._collapse_pipeline_params(
#         #     pipeline_steps,
#         #     self.parameters['pipeline']['params']
#         # )
#         #
#         # if "name" in self.parameters["model"].keys():
#         #
#         #     if "classification" in self.pipeline_folders and "regression" not in str(self.__class__):
#         #         model_module = importlib.import_module(
#         #             "django_learning.supervised.models.classification.{0}".format(self.parameters["model"]['name'])
#         #         )
#         #     elif "regression" in self.pipeline_folders or "regression" in str(self.__class__):
#         #         model_module = importlib.import_module(
#         #             "django_learning.supervised.models.regression.{0}".format(self.parameters["model"]['name'])
#         #         )
#         #     else:
#         #         model_module = importlib.import_module(
#         #             "django_learning.supervised.models.{0}".format(self.parameters["model"]['name'])
#         #         )
#         #
#         #     model_params = model_module.get_params()
#         #     model_class = model_params.pop("model_class")
#         #     model_params = model_params["params"]
#         #     pipeline_steps.append(("model", model_class))
#         #
#         #     params.update({"model__{}".format(k): v for k, v in model_params.iteritems()})
#         #     if 'params' in self.parameters['model'].keys():
#         #         params.update({"model__{}".format(k): v for k, v in self.parameters['model']['params'].iteritems()})
#         #
#         # hashstr = "".join([
#         #     self.cache_identifier,
#         #     str(OrderedDict(sorted(self.parameters.get("codes", {}).items(), key=lambda t: t[0]))),
#         #     str(OrderedDict(sorted(self.parameters.get("documents", {}).items(), key=lambda t: t[0]))),
#         #     str(get_pipeline_repr(pipeline_steps)),
#         #     str(get_param_repr(params)),
#         #     str(OrderedDict(sorted(self.parameters.get("model", {}).items(), key=lambda t: t[0])))
#         # ])
#         #
#         # cache_data = None
#         #
#         # if not refresh:
#         #     cache_data = self.cache.read(hashstr)
#         #
#         #     # cache_data['test_y'] = cache_data['test_y'].reset_index()
#         #     # del cache_data['test_y']['index']
#         #     # cache_data['test_x'] = cache_data['test_x'].reset_index()
#         #     # del cache_data['test_x']['index']
#         #     # cache_data['test_ids'] = cache_data['test_x'].index
#         #     # print "Resetting test indices"
#         #     # self.cache.write(hashstr, cache_data)
#         #
#         #     # if "test_x_old" not in cache_data.keys():
#         #     #     from logos.models import *
#         #     #     cache_data['test_x_old'] = cache_data['test_x']
#         #     #     cache_data['test_y_old'] = cache_data['test_y']
#         #     #     cache_data['test_ids_old'] = cache_data['test_ids']
#         #     #     cache_data['predict_y_old'] = cache_data['predict_y']
#         #     #     good_docs = DocumentSampleDocument.objects.filter(sample__in=DocumentSample.objects.filter(pk__in=[12, 22])).values_list("document_id", flat=True)
#         #     #     print "Old length: {}".format(len(cache_data['test_x']))
#         #     #     cache_data['test_x'] = cache_data['test_x'][cache_data['test_x']['document_id'].isin(good_docs)]
#         #     #     print "New length: {}".format(len(cache_data['test_x']))
#         #     #     cache_data['test_y'] = cache_data['test_y'].iloc[cache_data['test_x'].index]
#         #     #     cache_data['test_ids'] = cache_data['test_x'].index
#         #     #     cache_data['predict_y'] = pandas.Series(cache_data['predict_y']).iloc[cache_data['test_x'].index].values
#         #     #     self.cache.write(hashstr, cache_data)
#         #
#         # if is_null(cache_data) and not only_load_existing:
#         #
#         #     cache_data = CacheHandler("learning/supervised").read(hashstr)
#         #     if is_null(cache_data):
#         #         cache_data = self._get_model(pipeline_steps, params, **kwargs)
#         #     self.cache.write(hashstr, cache_data)
#         #
#         # if is_not_null(cache_data):
#         #     for k, v in cache_data.iteritems():
#         #         setattr(self, k, v)
#
#     #@require_training_data
#     def _get_model(self, pipeline_steps, params, **kwargs):
#
#         pass
#         # df = self.training_data
#         #
#         # smallest_code = df[self.outcome_variable].value_counts(ascending=True).index[0]
#         # largest_code = df[self.outcome_variable].value_counts(ascending=False).index[0]
#         #
#         # # print "Code frequencies: {}".format(dict(df[self.outcome_variable].value_counts(ascending=True)))
#         #
#         # if "training_weight" not in df.columns:
#         #     df["training_weight"] = 1.0
#         #
#         # if self.parameters["model"].get("use_class_weights", False):
#         #
#         #     class_weights = {}
#         #     base_weight = df[df[self.outcome_variable] == largest_code]['training_weight'].sum()
#         #     # total_weight = df['training_weight'].sum()
#         #     for c in df[self.outcome_variable].unique():
#         #         # class_weights[c] = float(df[df[self.outcome_variable]==c]["training_weight"].sum()) / float(total_weight)
#         #         class_weights[c] = base_weight / float(df[df[self.outcome_variable] == c]['training_weight'].sum())
#         #     total_weight = sum(class_weights.values())
#         #     class_weights = {k: float(v) / float(total_weight) for k, v in class_weights.items()}
#         #     params["model__class_weight"] = [class_weights, ]
#         #     print "Class weights: {}".format(class_weights)
#         #
#         # print "Creating train-test split"
#         #
#         # y = df[self.outcome_variable]
#         # X_cols = df.columns.tolist()
#         # X_cols.remove(self.outcome_variable)
#         # X = df[X_cols]
#         # if self.parameters["model"]["test_percent"] == 0.0:
#         #     X_train, X_test, y_train, y_test, train_ids, test_ids = X, None, y, None, y.index, None
#         #     print "Training on all {} cases".format(len(y_train))
#         # else:
#         #     X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(X, y, y.index, test_size=
#         #     self.parameters["model"]["test_percent"], random_state=5)
#         #     print "Selected %i training cases and %i test cases" % (
#         #         len(y_train),
#         #         len(y_test)
#         #     )
#         #
#         # scoring_function = None
#         # if "scoring_function" in self.parameters["model"].keys():
#         #     scoring_function = self._get_scoring_function(
#         #         self.parameters["model"]["scoring_function"],
#         #         binary_base_code=smallest_code if len(y.unique()) == 2 else None
#         #     )
#         #
#         # print "Beginning grid search using %s and %i cores for %s" % (
#         #     str(scoring_function),
#         #     self.num_cores,
#         #     self.outcome_variable
#         # )
#         #
#         # model = GridSearchCV(
#         #     Pipeline(pipeline_steps),
#         #     params,
#         #     fit_params={'model__sample_weight': [x for x in X_train["training_weight"].values]} if self.parameters[
#         #         "model"].get("use_sample_weights", False) else {},
#         #     cv=self.parameters["model"].get("cv", 5),
#         #     n_jobs=self.num_cores,
#         #     verbose=1,
#         #     scoring=scoring_function
#         # )
#         #
#         # model.fit(X_train, y_train)
#         #
#         # print "Finished training model, best score: {}".format(model.best_score_)
#         #
#         # predict_y = model.predict(X_test) if is_not_null(X_test) else None
#         #
#         # cache_data = {
#         #     "model": model,
#         #     "train_x": X_train,
#         #     "train_y": y_train,
#         #     "train_ids": train_ids,
#         #     "test_x": X_test,
#         #     "test_y": y_test,
#         #     "test_ids": test_ids,
#         #     "predict_y": predict_y
#         # }
#         #
#         # return cache_data
#
#     def _get_scoring_function(self, func_name, binary_base_code=None):
#
#         pass
#         # try:
#         #
#         #     from django_learning.utils import scoring_functions
#         #     scoring_function = scoring_functions[func_name]
#         #     # scoring_function = importlib.import_module(
#         #     #     "logos.learning.utils.scoring_functions.{0}".format(func_name)
#         #     # )
#         #     scoring_function = make_scorer(scoring_function.scorer)
#         #
#         # except:
#         #
#         #     if "regression" in str(self.__class__):
#         #         func_map = {
#         #             "mean_squared_error": (mean_squared_error, False, False),
#         #             "r2": (r2_score, True, False)
#         #         }
#         #         func, direction, needs_proba = func_map[func_name]
#         #         scoring_function = make_scorer(func, needs_proba=needs_proba, greater_is_better=direction)
#         #     elif binary_base_code:
#         #         func_map = {
#         #             "f1": (f1_score, True, False),
#         #             "precision": (precision_score, True, False),
#         #             "recall": (recall_score, True, False),
#         #             "brier_loss": (brier_score_loss, False, True)
#         #         }
#         #         func, direction, needs_proba = func_map[func_name]
#         #         scoring_function = make_scorer(func, needs_proba=needs_proba, greater_is_better=direction,
#         #                                        pos_label=binary_base_code)
#         #     else:
#         #         if self.parameters["model"]["scoring_function"] == "f1":
#         #             scoring_function = "f1_macro"
#         #             # scoring_function = "f1_micro"
#         #             # scoring_function = "f1_weighted"
#         #         elif self.parameters["model"]["scoring_function"] == "precision":
#         #             scoring_function = "precision"
#         #         else:
#         #             scoring_function = "recall"
#         #
#         # return scoring_function
#
#     def _collapse_pipeline_params(self, pipeline, params, names=None):
#
#         pass
#         # final_params = {}
#         # if not names:
#         #     names = []
#         # if isinstance(pipeline, Pipeline):
#         #     for sname, step in pipeline.steps:
#         #         final_params.update(self._collapse_pipeline_params(step, params, names=names + [sname]))
#         # elif isinstance(pipeline, FeatureUnion):
#         #     final_params.update(self._collapse_pipeline_params(pipeline.transformer_list, params, names=names))
#         # elif isinstance(pipeline, tuple):
#         #     final_params.update(pipeline[1], params, names=names + [pipeline[0]])
#         # elif isinstance(pipeline, list):
#         #     for sname, step in pipeline:
#         #         final_params.update(self._collapse_pipeline_params(step, params, names=names + [sname]))
#         # else:
#         #     if names[-1] in params.keys():
#         #         for k, v in params[names[-1]].iteritems():
#         #             # if k == "preprocessors":
#         #             #     preprocessor_sets = []
#         #             #     for pset in v:
#         #             #         preprocessors = []
#         #             #         try:
#         #             #             for preprocessor_name, preprocessor_params in pset:
#         #             #                 preprocessor_module = importlib.import_module("logos.learning.utils.preprocessors.{0}".format(preprocessor_name))
#         #             #                 preprocessors.append(preprocessor_module.Preprocessor(**preprocessor_params))
#         #             #         except ValueError: pass
#         #             #         preprocessor_sets.append(preprocessors)
#         #             #     v = preprocessor_sets
#         #             if len(v) > 0:
#         #                 final_params["__".join(names + [k])] = v
#         #     if isinstance(pipeline, BasicExtractor):
#         #         final_params["{}__cache_identifier".format("__".join(names))] = [self.cache_identifier]
#         #         final_params["{}__feature_name_prefix".format("__".join(names))] = [names[-1]]
#         #         if hasattr(self, 'document_types'):
#         #             final_params["{}__document_types".format("__".join(names))] = [self.document_types]
#         #
#         # return final_params
#
#     #@require_model
#     def get_feature_names(self, m):
#
#         pass
#         # features = []
#         #
#         # if hasattr(m, "steps"):
#         #     for name, step in m.steps:
#         #         features.append(self.get_feature_names(step))
#         # elif hasattr(m, "transformer_list"):
#         #     for name, step in m.transformer_list:
#         #         features.append(self.get_feature_names(step))
#         # elif hasattr(m, "get_feature_names"):
#         #     return m.get_feature_names()
#         #
#         # return [f for sublist in features for f in sublist]
#
#     #@require_model
#     def print_report(self):
#
#         pass
#         # print "'%s' results" % self.outcome_variable
#         #
#         # print "Best score: {} ({} std.)".format(self.model.best_score_,
#         #                                         getattr(self.model, "best_score_std_", None))
#         #
#         # # print "Best parameters:"
#         # # params = self.model.best_params_
#         # # for p in params.keys():
#         # #     if p.endswith("__stop_words"):
#         # #         del params[p]
#         # # print params
#         #
#         # try:
#         #     self.show_top_features()
#         # except:
#         #     pass
#
#     #@require_model
#     #@temp_cache_wrapper
#     def apply_model(self, data, keep_cols=None, clear_temp_cache=True):
#
#         pass
#         # if not keep_cols: keep_cols = []
#         #
#         # predictions = self.model.predict(data)
#         # try:
#         #     probabilities = self.model.predict_proba(data)
#         # except AttributeError:
#         #     probabilities = [None] * len(data)
#         #
#         # codes = []
#         # for index, pred, prob in zip(data.index, predictions, probabilities):
#         #     if type(prob) == list or type(prob) == tuple:
#         #         prob = max(prob)
#         #     code = {
#         #         self.outcome_variable: pred,
#         #         "probability": prob
#         #     }
#         #     for col in keep_cols:
#         #         code[col] = data.loc[index, col]
#         #     codes.append(code)
#         #
#         # return pandas.DataFrame(codes)
#
#     # TODO: convert everything to model-based handlers
#     # move everything from BasicHandler, etc, over to these
#     # you should still store the big stuff in S3, but it'll be cleaner and easier to modify and update
#     # if you don't have separate handler classes
#
#
# class DocumentLearningModel(LearningModel):
#
#     document_types = ArrayField(models.CharField(max_length=60, choices=get_document_types()),
#                                 help_text="The type of documents the classifier was trained to code")
#     # frame = models.ForeignKey("django_learning.DocumentSampleFrame", related_name="classifiers", null=True)
#     frames = models.ManyToManyField("django_learning.DocumentSampleFrame", related_name="classifiers")
#
#     objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()
#
#     class Meta:
#
#         abstract = True
#
#
# class DocumentClassificationModel(DocumentLearningModel):
#
#     """
#     Holds data for a classifier for a given variable and type of document.  Only one classifier may exist for a given
#     combination of code variable and document type - to keep things simple and uncomplicated (we don't want competing
#     alternative versions of a model that do the same thing.)  Holds pickled objects that contain the training/test data
#     and parameters used to train the model, and the actual trained model itself.
#     """
#
#     variable = models.ForeignKey("django_learning.CodeVariable", related_name="classifiers",
#                                  help_text="The code variable whose codes the classifier is attempting to assign")
#
#     objects = DJANGO_LEARNING_BASE_MANAGER().as_manager()
#
#     def __str__(self):
#
#         return "{}, {}, {}".format(self.name, self.variable.name, self.document_types)
#
#     def save(self, *args, **kwargs):
#
#         self.parameters['pipeline']['steps'] = [(k, v) for k, v in self.parameters['pipeline']['steps'] if k != 'model']
#         # TODO: figure out where this bug is occuring, but for now we'll deal with extra "model" params sneaking in
#
#         super(DocumentClassificationModel, self).save(*args, **kwargs)
#
#     @property
#     def handler(self):
#
#         handler_class = None
#         module = importlib.import_module("django_learning.learning.supervised")
#         if hasattr(module, self.handler_class):
#             handler_class = getattr(module, self.handler_class)
#         if handler_class:
#             if self.handler_class == "MetaDocumentClassificationHandler":
#                 h = handler_class(
#                     self.variable.name,
#                     self.parameters["classifiers"],
#                     self.parameters["operator"],
#                     saved_name=self.name,
#                     verbose=False
#                 )
#             else:
#                 h = handler_class(
#                     self.document_types,
#                     self.variable.name,
#                     saved_name=self.name,
#                     verbose=False
#                 )
#
#         return h
#
#         # if self.parameters["model"].get("binary_scoring_function"):
#         #     h = DocumentClassificationRegressionHandler(
#         #         self.document_types,
#         #         self.variable.name,
#         #         saved_name=self.name,
#         #         verbose=False
#         #     )
#         # elif "classifiers" in self.parameters.keys():
#         #     h = MetaDocumentClassificationHandler(
#         #         self.variable.name,
#         #         self.parameters["classifiers"],
#         #         self.parameters["operator"]
#         #     )
#         # else:
#         #     h = DocumentClassificationHandler(
#         #         self.document_types,
#         #         self.variable.name,
#         #         saved_name=self.name,
#         #         verbose=False
#         #     )
#         # return h
#
#     @property
#     def training_data(self):
#
#         return self.handler.training_data
#
#     @property
#     def model(self):
#
#         return self.handler.model
#
#     def compute_cv_folds(self, use_test_data=False, num_folds=5, refresh=False):
#
#         h = self.handler
#         fold_preds = None
#         if (not use_test_data and (is_null(self.cv_folds) or refresh)) or \
#                 (use_test_data and (is_null(self.cv_folds_test) or refresh)):
#             fold_preds = h.compute_cv_folds(use_test_data=use_test_data, num_folds=num_folds)
#         if fold_preds:
#             if use_test_data:
#                 self.cv_folds_test = fold_preds
#             else:
#                 self.cv_folds = fold_preds
#             self.save()
#
#     def get_code_cv_training_scores(self, use_test_data=False, code_value="1", partition_by=None,
#                                     restrict_document_type=None, min_support=0):
#
#         h = self.handler
#         h.load_model(only_load_existing=True)
#         if is_not_null(h.model):
#
#             if use_test_data:
#                 X = h.test_x
#                 y = h.test_y
#                 fold_preds = self.cv_folds_test
#             else:
#                 X = h.train_x
#                 y = h.train_y
#                 fold_preds = self.cv_folds
#
#             return h.get_code_cv_training_scores(
#                 fold_preds, X, y,
#                 code_value=code_value,
#                 partition_by=partition_by,
#                 restrict_document_type=restrict_document_type,
#                 min_support=min_support
#             )
#
#         else:
#             print "Couldn't find cached model to load using the saved parameters"
#             return None
#
#     def get_code_validation_test_scores(self, code_value="1", partition_by=None, restrict_document_type=None,
#                                         use_expert_consensus_subset=False, compute_for_experts=False, min_support=0):
#
#         h = self.handler
#         h.load_model(only_load_existing=True)
#         if is_not_null(h.model):
#
#             return h.get_code_validation_test_scores(
#                 code_value=code_value,
#                 partition_by=partition_by,
#                 restrict_document_type=restrict_document_type,
#                 use_expert_consensus_subset=use_expert_consensus_subset,
#                 compute_for_experts=compute_for_experts,
#                 min_support=min_support
#             )
#
#         else:
#             print "Couldn't find cached model to load using the saved parameters"
#             return None
#
#     def show_top_features(self, n=10):
#
#         self.handler.show_top_features(n=n)
#
#     def print_report(self):
#
#         self.handler.print_report()
#
#     def get_report_results(self):
#
#         return self.handler.get_report_results()
#
#     def apply_model_to_frames(self, num_cores=2, chunk_size=1000, refresh_existing=False):
#
#         h = self.handler
#         h.load_model(only_load_existing=True)
#         if is_not_null(h.model):
#             print "Selecting frame documents"
#             docs = get_model("Document").objects.filter(sample_frames__in=self.frames.all())
#             if not refresh_existing:
#                 existing = self.coded_documents.values_list("document_id", flat=True)
#                 keep = get_model("Document").objects.filter(
#                     pk__in=set(docs.values_list("pk", flat=True)).difference(set(existing)))
#                 print "Skipping {} existing documents, {} remaining".format(existing.count(), keep.count())
#                 # if existing.count() > 0:
#                 #    docs = docs.exclude(pk__in=existing)
#                 docs = keep
#             print "Applying model to {} documents".format(docs.count())
#             h.apply_model_to_database(docs, chunk_size=chunk_size, num_cores=num_cores)
#
#
# class DocumentRegressionModel(DocumentLearningModel):
#
#     pass
