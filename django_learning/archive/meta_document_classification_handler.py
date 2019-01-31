# import itertools, numpy, pandas, copy, importlib
#
# from django.db.models import Q
# from nltk.metrics.agreement import AnnotationTask
# from collections import defaultdict
# from sklearn.metrics import classification_report
# from sklearn.cross_validation import cross_val_predict
# from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, brier_score_loss
# from statsmodels.stats.inter_rater import cohens_kappa
# from sklearn.dummy import DummyClassifier
# from scipy.stats import ttest_ind
# from tqdm import tqdm
# from multiprocessing import Pool
#
# from logos.learning.utils.decorators import require_model, require_training_data, temp_cache_wrapper
# from logos.utils import is_not_null, is_null
# from logos.utils.data import chunk_list, compute_sample_weights_from_frame, compute_balanced_sample_weights, wmom
# from logos.utils.database import get_model
# from logos.utils.text import decode_text
# from .classification_handler import ClassificationHandler
# from .document_classification_handler import DocumentClassificationHandler
#
#
# class MetaDocumentClassificationHandler(DocumentClassificationHandler):
#
#     def __init__(self,
#         code_variable_name,
#         classifier_names,
#         operator,
#         saved_name=None,
#         **kwargs
#     ):
#
#         self.classifiers = get_model("DocumentClassificationModel").objects.filter(name__in=classifier_names)
#
#         var = get_model("CodeVariable").objects.get_if_exists({"name": code_variable_name})
#         if not var:
#             print "Creating new meta-variable {}".format(code_variable_name)
#             var = get_model("CodeVariable").objects.create_or_update({"name": code_variable_name})
#             for binary_val, label in [("1", "Yes"), ("0", "No")]:
#                 if binary_val not in var.codes.values_list("value", flat=True):
#                     print "Creating new code value '{}'".format(binary_val)
#                     get_model("Code").objects.create_or_update(
#                         {"variable": var, "value": binary_val},
#                         {"label": label},
#                         return_object=False
#                     )
#
#         doc_types = []
#         for c in self.classifiers.all(): doc_types.extend(c.document_types)
#
#         super(MetaDocumentClassificationHandler, self).__init__(
#             list(set(doc_types)),
#             code_variable_name,
#             saved_name=saved_name,
#             params={
#                 "codes": {},
#                 "documents": {
#                     "frames": get_model("DocumentSampleFrame").objects.filter(classifiers__in=self.classifiers.all()).values_list("name", flat=True)
#                 },
#                 "pipeline": {"steps": [], "params": {}},
#                 "model": {},
#                 "classifiers": classifier_names,
#                 "operator": operator
#             },
#             **kwargs
#         )
#
#         self.operator = operator
#
#         # if get_model("CoderDocumentCode").objects.filter(code__variable=self.code_variable).count() == 0:
#         #     print "No meta-variable codes exist for {}, computing".format(code_variable_name)
#         #     self._sync_meta_variable()
#
#
#     def _sync_meta_variable(self):
#
#         filtered_docs = {}
#         for c in self.classifiers.all():
#             h = c.handler
#             h.load_model()
#             filtered_docs[c.name] = list(set(list(h.test_x['pk'].values) + list(h.train_x['pk'].values)))
#
#         docs = get_model("Document").objects.filter(sampling_frames__in=self.frames.all())
#         sample_units = get_model("DocumentSampleDocument").objects.filter(document__in=docs)
#         for su in tqdm(sample_units, desc="Syncing meta-variable codes", total=sample_units.count()):
#             coder_ids = get_model("CoderDocumentCode").objects.filter(sample_unit=su).values_list("coder_id", flat=True)
#             for coder in get_model("Coder").objects.filter(pk__in=coder_ids):
#                 vals = []
#                 for c in self.classifiers.all():
#                     code = coder.coded_documents.get_if_exists({"code__variable": c.variable, "sample_unit": su})
#                     if code:
#                         if su.document_id not in filtered_docs[c.name]:
#                             vals.append(0)
#                         else:
#                             vals.append(int(code.code.value))
#                 val = None
#                 if len(vals) == self.classifiers.count():
#                     if self.operator == "any":
#                         val = 1 if any([v == 1 for v in vals]) else 0
#                     elif self.operator == "all":
#                         val = 1 if all([v == 1 for v in vals]) else 0
#                 if val != None:
#                     get_model("CoderDocumentCode").objects.create_or_update(
#                         {"code": self.code_variable.codes.get(value=str(val)), "coder": coder, "sample_unit": su},
#                         {"hit_id": code.hit_id},
#                         return_object=None
#                     )
#
#     def _get_training_data(self, validation=False):
#
#         return {}
#
#     @require_training_data
#     def _get_model(self, pipeline_steps, params):
#
#         handlers, indices, output_dfs = {}, defaultdict(list), {}
#         for c in self.classifiers.all():
#             h = c.handler
#             h.load_model()
#             handlers[c.name] = h
#             indices["train"].extend(h.train_x['document_id'])
#             indices["test"].extend(h.test_x['document_id'])
#             indices["predict"].extend(h.test_x['document_id'])
#         for df_name in ["train", "test", "predict"]:
#             output_dfs[df_name] = pandas.DataFrame(columns=[c.name for c in self.classifiers], index=list(set(indices[df_name])))
#         for c_name, h in handlers.iteritems():
#             input_x, input_y = {}, {}
#             input_x["train"] = h.train_x
#             input_x["test"] = h.test_x
#             input_x["predict"] = h.test_x
#             input_y["train"] = h.train_y
#             input_y["test"] = h.test_y
#             input_y["predict"] = pandas.Series(h.predict_y)
#             code_map = {c.pk: c.value for c in h.code_variable.codes.all()}
#             for df_name in ["train", "test", "predict"]:
#                 input_y[df_name].index = input_x[df_name]['document_id']
#                 input_x[df_name].index = input_x[df_name]['document_id']
#                 output_dfs[df_name][c_name].ix[input_y[df_name].index] = input_y[df_name].map(code_map)
#                 if "{}_weight".format(c_name) not in output_dfs[df_name].columns:
#                     output_dfs[df_name]["{}_weight".format(c_name)] = None
#                 output_dfs[df_name]["{}_weight".format(c_name)].ix[input_y[df_name].index] = input_x[df_name]["sampling_weight"]
#
#         for df_name in ["train", "test", "predict"]:
#             output_dfs[df_name]["sampling_weight"] = output_dfs[df_name].apply(lambda x: numpy.average([x['{}_weight'.format(c.name)] for c in self.classifiers.all() if x['{}_weight'.format(c.name)]]), axis=1)
#             # for c in self.classifiers.all():
#             #     output_dfs[df_name][c.name] = output_dfs[df_name][c.name].fillna(0)
#             if self.operator == "any":
#                 for c in self.classifiers.all():
#                     output_dfs[df_name][c.name] = output_dfs[df_name][c.name].fillna(0)
#                 output_dfs[df_name][self.outcome_variable] = output_dfs[df_name].apply(
#                     lambda x: 1 if any([int(x[c.name])==1 for c in self.classifiers.all()]) else 0, axis=1)
#             elif self.operator == "all":
#                 output_dfs[df_name] = output_dfs[df_name].dropna(subset=[c.name for c in self.classifiers.all()])
#                 output_dfs[df_name][self.outcome_variable] = output_dfs[df_name].apply(
#                     lambda x: 1 if all([int(x[c.name])==1 for c in self.classifiers.all()]) else 0, axis=1)
#             for c in self.classifiers.all():
#                 del output_dfs[df_name][c.name]
#                 del output_dfs[df_name]["{}_weight".format(c.name)]
#
#         for k in output_dfs.keys():
#             output_dfs[k]['pk'] = output_dfs[k].index
#             output_dfs[k]['document_id'] = output_dfs[k].index
#             output_dfs[k] = output_dfs[k].reset_index()
#             code_map = {}
#             for code_id in output_dfs[k][self.outcome_variable].unique():
#                 code_map[code_id] = self.code_variable.codes.get(value=code_id).pk
#             output_dfs[k][self.outcome_variable] = output_dfs[k][self.outcome_variable].map(code_map)
#
#         results = {
#             "predict_y": output_dfs["predict"][self.outcome_variable],
#             "test_y": output_dfs["test"][self.outcome_variable],
#             "train_y": output_dfs["train"][self.outcome_variable],
#             "predict_x": output_dfs["predict"][["pk", "document_id", "sampling_weight"]],
#             "test_x": output_dfs["test"][["pk", "document_id", "sampling_weight"]],
#             "train_x": output_dfs["train"][["pk", "document_id", "sampling_weight"]],
#             "test_ids": output_dfs["test"].index,
#             "train_ids": output_dfs["train"].index,
#             "model": "PLACEHOLDER"
#         }
#
#         return results
#
#
#     @require_model
#     def apply_model(self, documents, keep_cols=None):
#
#         if not keep_cols: keep_cols = ["pk"]
#         documents = pandas.DataFrame.from_records(documents.values(*keep_cols))
#         if len(documents) > 0:
#             for c in self.classifiers.all():
#                 h = c.handler
#                 h.load_model()
#                 filtered_docs = h.filter_documents(get_model("Document").objects.filter(pk__in=documents['pk'].values))
#                 predictions = {}
#                 for index, row in filtered_docs.iterrows():
#                     try:
#                         code = get_model("ClassifierDocumentCode").objects.get(
#                             classifier=c,
#                             document_id=row['pk']
#                         )
#                         predictions[row['pk']] = code.code.value
#                     except get_model("ClassifierDocumentCode").DoesNotExist:
#                         pass
#                 remaining = filtered_docs[~filtered_docs['pk'].isin(predictions.keys())]
#                 if len(remaining) > 0:
#                     remaining[h.outcome_variable] = h.model.predict(remaining)
#                     for index, row in remaining.iterrows():
#                         predictions[row['pk']] = get_model("Code").objects.get(pk=row[h.outcome_variable]).value
#                 documents[c.name] = documents['pk'].map(lambda x: predictions.get(x, 0))
#
#             if self.operator == "any":
#                 documents[self.outcome_variable] = documents.apply(
#                     lambda x: 1 if any([int(x[c.name]) == 1 for c in self.classifiers.all()]) else 0, axis=1)
#             elif self.operator == "all":
#                 documents[self.outcome_variable] = documents.apply(
#                     lambda x: 1 if all([int(x[c.name]) == 1 for c in self.classifiers.all()]) else 0, axis=1)
#
#             for col in documents.columns:
#                 if col not in ["pk", self.outcome_variable] + keep_cols:
#                     del documents[col]
#
#             code_map = {}
#             for code_id in documents[self.outcome_variable].unique():
#                 code_map[code_id] = self.code_variable.codes.get(value=code_id).pk
#             documents[self.outcome_variable] = documents[self.outcome_variable].map(code_map)
#
#             documents['probability'] = None
#
#             return documents
#
#         else:
#
#             return pandas.DataFrame()
#
#
#     @require_model
#     def compute_cv_folds(self, use_test_data=False, num_folds=5, folds=None):
#
#         if use_test_data:
#             train_x = self.test_x[~self.test_x['document_id'].isin(self.train_x['document_id'])]
#             train_y = self.test_y.ix[train_x.index]
#         else:
#             train_x = self.train_x
#             train_y = self.train_y
#
#         if not folds:
#
#             final_model = DummyClassifier()
#             fold_preds = cross_val_predict(final_model, train_x, train_y,
#                 keep_folds_separate=True,
#                 cv=num_folds
#             )
#             folds = []
#             for fold, preds in enumerate(fold_preds):
#                 preds, indices = preds
#                 folds.append(train_x['pk'][indices].values)
#
#         fold_preds = defaultdict(list)
#         for c in self.classifiers.all():
#
#             h = c.handler
#             h.load_training_data()
#             h.load_model()
#
#             if use_test_data:
#                 c_train_x = h.test_x
#                 c_train_y = h.test_y
#             else:
#                 c_train_x = h.train_x
#                 c_train_y = h.train_y
#
#             if c.handler_class == "MetaDocumentClassificationHandler":
#
#                 preds = h.compute_cv_folds(use_test_data=use_test_data, num_folds=num_folds, folds=folds)
#                 fold_preds[c.name] = [(c_train_x['pk'][indices].values, predset) for predset, indices in preds]
#
#             else:
#
#                 final_model = h.model.best_estimator_
#
#                 for fold in folds:
#                     train_indices = c_train_x[~c_train_x['pk'].isin(fold)].index
#                     test_indices = c_train_x[c_train_x['pk'].isin(fold)].index
#                     final_model = final_model.fit(
#                         c_train_x.iloc[train_indices],
#                         c_train_y.iloc[train_indices],
#                         model__sample_weight=[x for x in c_train_x["sampling_weight"][train_indices].values] if h.parameters["model"].get("use_sample_weights", False) else None
#                     )
#                     preds = final_model.predict(c_train_x.iloc[test_indices])
#                     fold_preds[c.name].append((c_train_x['pk'][test_indices].values, preds))
#
#         val_map = {v['pk']: int(v['value']) for v in self.code_variable.codes.values("pk", "value")}
#         for c in self.classifiers.all():
#             val_map.update({v['pk']: int(v['value']) for v in c.variable.codes.values("pk", "value")})
#         val_map[None] = None
#         pk_map = {int(v['value']): v['pk'] for v in self.code_variable.codes.values("pk", "value")}
#         # for c in self.classifiers.all():
#         #     pk_map.update({int(v['value']): v['pk'] for v in c.variable.codes.values("pk", "value")})
#         final_folds = []
#         for i in range(0, len(folds)):
#             preds = defaultdict(dict)
#             for c in self.classifiers.all():
#                 for index, pred in zip(*fold_preds[c.name][i]):
#                     preds[index][c.name] = pred
#             indices, final_preds, pks = [], [], []
#             for index in preds.keys():
#                 indices.append(train_x[train_x['pk']==index].index[0])
#                 pks.append(index)
#                 if self.operator == "any":
#                     if any([val_map[preds[index].get(c.name, None)]==1 for c in self.classifiers.all()]):
#                         final_preds.append(pk_map[1])
#                     else:
#                         final_preds.append(pk_map[0])
#                 elif self.operator == "all":
#                     if all([val_map[preds[index].get(c.name, None)]==1 for c in self.classifiers.all()]):
#                         final_preds.append(pk_map[1])
#                     else:
#                         final_preds.append(pk_map[0])
#             final_folds.append((numpy.array(final_preds), numpy.array(indices)))
#
#         return final_folds
#
#
#     def _get_expert_consensus_dataframe(self, code_value, coders=None, use_consensus_ignore_flag=True):
#
#         merged_df = None
#         for c in self.classifiers.all():
#
#             h = c.handler
#             h.load_model()
#             df = h._get_expert_consensus_dataframe(code_value, coders=coders, use_consensus_ignore_flag=use_consensus_ignore_flag)
#             df = df[["code_id", "code", "sampling_weight", "document_id"]].rename(
#                 columns={
#                     "code_id": "{}_code_id".format(c.name),
#                     "code": "{}_code".format(c.name),
#                     "sampling_weight": "{}_sampling_weight".format(c.name)
#                 }
#             )
#             if is_null(merged_df):
#                 merged_df = df
#             else:
#                 merged_df = pandas.merge(merged_df, df, how='outer', on='document_id')
#
#         merged_df['sampling_weight'] = merged_df.apply(lambda x: numpy.average([x['{}_sampling_weight'.format(c.name)] for c in self.classifiers.all() if is_not_null(x['{}_sampling_weight'.format(c.name)])]), axis=1)
#         if self.operator == "any":
#             for c in self.classifiers.all():
#                 merged_df["{}_code".format(c.name)] = merged_df["{}_code".format(c.name)].fillna(0)
#             merged_df['code'] = merged_df.apply(lambda x: 1 if any([int(x['{}_code'.format(c.name)]) == 1 for c in self.classifiers.all()]) else 0, axis=1)
#         elif self.operator == "all":
#             merged_df = merged_df.dropna(subset=["{}_code".format(c.name) for c in self.classifiers.all()])
#             merged_df['code'] = merged_df.apply(lambda x: 1 if all([int(x['{}_code'.format(c.name)]) == 1 for c in self.classifiers.all()]) else 0, axis=1)
#         code_map = {int(c.value): c.pk for c in self.code_variable.codes.all()}
#         merged_df['code_id'] = merged_df['code'].map(code_map)
#         merged_df = merged_df[['document_id', 'code_id', 'code', 'sampling_weight']]
#
#         return merged_df