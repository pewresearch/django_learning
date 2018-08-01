# import itertools, numpy, pandas, copy, importlib
#
# from django.db.models import Q
# from nltk.metrics.agreement import AnnotationTask
# from sklearn.metrics import classification_report
# from statsmodels.stats.inter_rater import cohens_kappa
# from collections import defaultdict
#
# from logos.learning.utils.decorators import require_model, require_training_data, temp_cache_wrapper
# from logos.utils import is_not_null, is_null
# from logos.utils.data import chunker, compute_sample_weights_from_frame, compute_balanced_sample_weights
# from logos.utils.database import get_model
# from logos.utils.text import decode_text
# from .classification_handler import ClassificationHandler
#
#
# class DocumentClassificationRegressionHandler(ClassificationHandler):
#
#     pipeline_folders = ["classification", "documents"]
#
#     def __init__(self,
#         document_types,
#         code_variable_name,
#         saved_name=None,
#         pipeline=None,
#         params=None,
#         num_cores=2,
#         verbose=True,
#         doc_limit=None,
#         **kwargs
#     ):
#         self.doc_limit = doc_limit
#
#         if isinstance(document_types, str):
#             document_types = [document_types]
#         else:
#             document_types = sorted(document_types)
#
#         super(DocumentClassificationRegressionHandler, self).__init__(
#             "_".join(["_".join(document_types), code_variable_name]),
#             outcome_variable="code_id",
#             pipeline=pipeline,
#             params=params,
#             num_cores=num_cores,
#             verbose=verbose,
#             **kwargs
#         )
#
#         self._document_types = document_types
#         self._code_variable = get_model("CodeVariable").objects.get(name=code_variable_name)
#         self._frames = None
#
#         if saved_name:
#
#             self.saved_model = get_model("DocumentClassificationModel").objects.get_if_exists({"name": saved_name})
#             if self.saved_model:
#                 if verbose:
#                     print "You passed the name of a saved CodeVaraibleClassifier record; all parameters passed (except for optional ones) will be overridden by the saved values in the database"
#                     if sorted(self.saved_model.document_types) != document_types:
#                         print "Document types overridden from saved database value: {0} to {1}".format(document_types, self.saved_model.document_types)
#                     if self.saved_model.variable.name != self._code_variable.name:
#                         print "Code variable overridden from saved database value: {0} to {1}".format(self._code_variable.name, self.saved_model.variable.name)
#                     if self.saved_model.pipeline_name != pipeline and is_not_null(pipeline):
#                         print "Named pipeline overridden from saved database value: {0} to {1}".format(pipeline, self.saved_model.pipeline_name)
#                 self._document_types = self.saved_model.document_types
#                 self._code_variable = self.saved_model.variable
#                 self._frames = self.saved_model.frames.all()
#                 self._parameters = self.saved_model.parameters
#                 self.pipeline_name = self.saved_model.pipeline_name
#             else:
#                 print "No classifier '{0}' found in the database".format(saved_name)
#
#         else:
#
#             self.saved_model = None
#
#         if not self._frames:
#             if "frames" in self.parameters["documents"].keys():
#                 self._frames = get_model("DocumentSampleFrame").objects.filter(name__in=self.parameters["documents"]["frames"])
#             else:
#                 self._frames = get_model("DocumentSampleFrame").objects.filter(name__in=[])
#
#         if self.saved_model and verbose:
#             print "Currently associated with a saved regressor: {0}".format(self.saved_model)
#
#         # Always pre-compute unnecessary/bad code parameters and delete them, to keep cache keys consistent:
#
#         def has_all_params(p):
#             return all([k in p.keys() for k in [
#                 "dataset_code_filters"
#             ]])
#
#         has_experts = self._has_raw_codes(turk=False)
#         self.use_expert_codes = False
#         if "experts" in self.parameters["codes"].keys():
#             if is_not_null(self.parameters["codes"]["experts"]) and has_all_params(self.parameters["codes"]["experts"]):
#                 if has_experts:
#                     self.use_expert_codes = True
#                 else: del self.parameters["codes"]["experts"]
#             else: del self.parameters["codes"]["experts"]
#
#         has_mturk = self._has_raw_codes(turk=True)
#         self.use_mturk_codes = False
#         if "mturk" in self.parameters["codes"].keys():
#             if is_not_null(self.parameters["codes"]["mturk"]) and has_all_params(self.parameters["codes"]["mturk"]):
#                 if has_mturk:
#                     self.use_mturk_codes = True
#                 else: del self.parameters["codes"]["mturk"]
#             else: del self.parameters["codes"]["mturk"]
#
#         fallback = False
#         if self.use_expert_codes and self.use_mturk_codes:
#             if not has_all_params(self.parameters["codes"]):
#                 fallback = True
#                 for k in ["dataset_code_filters", "mturk_to_expert_weight"]:
#                     if k in self.parameters["codes"]:
#                         del self.parameters["codes"][k]
#         else:
#             for k in ["dataset_code_filters",]:
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
#
#     @property
#     def parameters(self):
#         if self.saved_model: return self.saved_model.parameters
#         else: return self._parameters
#
#
#     @property
#     def code_variable(self):
#         # Wrapping this and document_types/parameters as properties ensures that handlers associated with a saved classifier make it difficult to modify these values once saved
#         if self.saved_model: return self.saved_model.variable
#         else: return self._code_variable
#
#
#     @property
#     def document_types(self):
#         if self.saved_model: return self.saved_model.document_types
#         else: return self._document_types
#
#
#     @property
#     def frames(self):
#         if self.saved_model: return self.saved_model.frames.all()
#         else: return self._frames
#
#
#     def _get_training_data(self, validation=False):
#
#         expert_codes = None
#         if self.use_expert_codes:
#
#             print "Extracting expert codes"
#             expert_codes = self._get_raw_codes(turk=False, training=(not validation))
#             print "{} expert codes extracted".format(len(expert_codes))
#
#             for filter_name, filter_params in self.parameters["codes"]["experts"]["dataset_code_filters"]:
#                 if is_not_null(expert_codes, empty_lists_are_null=True):
#                     print "Applying expert code filter: %s" % filter_name
#                     filter_module = importlib.import_module("logos.learning.utils.dataset_code_filters.{0}".format(filter_name))
#                     expert_codes = filter_module.filter(expert_codes, **filter_params)
#
#             if is_not_null(expert_codes):
#
#                 expert_codes = self._consolidate_codes(
#                     expert_codes,
#                     fake_coder_id="experts"
#                 )
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
#                 for filter_name, filter_params in self.parameters["codes"]["mturk"]["dataset_code_filters"]:
#                     print "Applying MTurk code filter: %s" % filter_name
#                     filter_module = importlib.import_module("logos.learning.utils.dataset_code_filters.{0}".format(filter_name))
#                     mturk_codes = filter_module.filter(mturk_codes, **filter_params)
#
#                 if is_not_null(mturk_codes):
#
#                     mturk_codes = self._consolidate_codes(
#                         mturk_codes,
#                         fake_coder_id="mturk"
#                     )
#
#         if self.use_expert_codes and self.use_mturk_codes:
#
#             codes = pandas.concat([c for c in [expert_codes, mturk_codes] if is_not_null(c)])
#
#             for filter_name, filter_params in self.parameters["codes"]["dataset_code_filters"]:
#                 print "Applying global code filter: %s" % filter_name
#                 filter_module = importlib.import_module("logos.learning.utils.dataset_code_filters.{0}".format(filter_name))
#                 codes = filter_module.filter(codes, **filter_params)
#
#             print "Consolidating all codes"
#
#             if "mturk_to_expert_weight" in self.parameters["codes"]:
#                 df = self._consolidate_codes(
#                     codes,
#                     mturk_to_expert_weight=self.parameters["codes"]["mturk_to_expert_weight"]
#                 )
#             else:
#                 df = self._consolidate_codes(
#                     codes
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
#         # TODO: get rid of this doc_limit stuff, it's just for testing
#         if self.doc_limit:
#             training_data = training_data.iloc[range(0, self.doc_limit)]
#         return {
#             "training_data": training_data
#         }
#
#
#     def _get_code_query(self, turk=False, training=False):
#
#         query = get_model("CoderDocumentCode").objects
#
#         if len(self.frames) > 0:
#             frame_query = Q()
#             for f in self.frames:
#                 frame_query.add(Q(**{"sample_unit__sample__frame": f}), Q.OR)
#             query = query.filter(frame_query)
#
#         filter_query = Q()
#         for doc_type in self.document_types:
#             filter_query.add(Q(**{"sample_unit__document__{}_id__isnull".format(doc_type): False}), Q.OR)
#
#         return query \
#             .filter(filter_query) \
#             .filter(code__variable=self.code_variable) \
#             .filter(coder__is_mturk=turk) \
#             .filter(sample_unit__sample__training=training)
#
#     def _has_raw_codes(self, turk=False):
#
#         return self._get_code_query(turk=turk, training=True).count() > 0
#
#     def _get_raw_codes(self, turk=False, training=True):
#
#         query = self._get_code_query(turk=turk, training=training)
#         columns = [
#             "sample_unit__document_id",
#             "code__value",
#             "coder_id",
#             "sample_unit__weight"
#         ]
#         for doc_type in self.document_types:
#             columns.append("sample_unit__document__{}_id".format(doc_type))
#         codes = pandas.DataFrame(list(query.values(*columns)))
#         if len(codes) > 0:
#             codes = codes.rename(columns={"sample_unit__weight": "sampling_weight", "sample_unit__document_id": "document_id"})
#             codes["sampling_weight"] = codes["sampling_weight"].fillna(1.0)
#             codes["is_mturk"] = turk
#             for doc_type in self.document_types:
#                 codes.ix[~codes["sample_unit__document__{}_id".format(doc_type)].isnull(), "document_type"] = doc_type
#             codes["code_id"] = codes["code__value"].map(lambda x: 1 if x == self.parameters["codes"]["class_value"] else 0)
#             del codes["code__value"]
#             # codes = pandas.concat([codes, pandas.get_dummies(codes["code_id"], prefix="code")], axis=1)
#             # codes = pandas.get_dummies(codes, prefix="code", columns=["code_id"])
#         else:
#             codes = None
#         if not self.parameters["documents"].get("include_sample_weights", False):
#             codes["sampling_weight"] = 1.0
#         codes["training_weight"] = codes["sampling_weight"]
#
#         return codes
#
#
#     def _consolidate_codes(self, codes, mturk_to_expert_weight=1.0, keep_quantile=None, fake_coder_id="consolidated"):
#
#         def _consolidate(doc_id, group):
#
#             has_mturk = True in group["is_mturk"].unique()
#             has_experts = False in group["is_mturk"].unique()
#
#             avgs = {}
#
#             if has_mturk and has_experts:
#
#                 turkers = group[group["is_mturk"]==True]["coder_id"].unique()
#                 experts = group[group["is_mturk"]==False]["coder_id"].unique()
#                 total_coders = float(len(turkers) + len(experts))
#
#                 expert_ratio = float(len(experts)) / total_coders
#                 adj_expert_ratio = expert_ratio * mturk_to_expert_weight
#                 adj_turk_ratio = 1.0 - min([1.0, adj_expert_ratio])
#
#                 avg = (adj_turk_ratio * group[group["coder_id"].isin(turkers)]["code_id"].mean()) + \
#                     (adj_expert_ratio * group[group["coder_id"].isin(experts)]["code_id"].mean())
#
#             else:
#
#                 avg = round(group["code_id"].mean(), 3)
#
#             avgs["code_id"] = avg
#
#             consolidated_row = {
#                 "coder_id": fake_coder_id,
#                 "code_id": avgs["code_id"],
#                 "document_type": group["document_type"].iloc[0],
#                 "document_id": doc_id,
#                 "is_mturk": (has_mturk and not has_experts),
#                 "sampling_weight": group['sampling_weight'].mean() if 'sampling_weight' in group.keys() else None,
#                 "training_weight": group['training_weight'].mean() if 'training_weight' in group.keys() else None
#             }
#
#             df = pandas.DataFrame([consolidated_row])
#             return df
#
#         new_df = pandas.concat([_consolidate(i, group) for i, group in codes.groupby("document_id")])
#
#         return new_df
#
#
#     def filter_documents(self, documents, additional_columns=None):
#
#         for filter_name, filter_params in self.parameters["documents"]["filters"]:
#             #print "Applying document filter: %s" % filter_name
#             filter_module = importlib.import_module("logos.learning.utils.dataset_document_filters.{0}".format(filter_name))
#             documents = filter_module.filter(documents, **filter_params)
#             # documents = getattr(document_filters, f)(documents)
#
#         if not additional_columns: additional_columns = []
#         df = pandas.DataFrame(list(documents.values("pk", "text", "date", *additional_columns)))
#         df['text'] = df['text'].apply(lambda x: decode_text(x))
#         df = df.dropna(subset=['text'])
#         df['text'] = df['text'].astype(str)
#
#         # This should now be happening when codes are first extracted
#         # if len(self.frames) > 0:
#         #     df = df[df['pk'].isin(self.frame.documents.values_list("pk", flat=True))]
#
#         return df
#
#
#     def _add_frame_weights(self, training_data):
#
#         # print "Adding sample weights"
#
#         # base_weights = self.code_variable.samples\
#         #     .filter(training=True)\
#         #     .filter(document_weights__document_id__in=training_data["pk"])\
#         #     .values("document_weights__document_id", "document_weights__weight")
#         # base_weights = pandas.DataFrame.from_records(base_weights)
#         # base_weights = base_weights.rename(columns={
#         #     "document_weights__document_id": "pk",
#         #     "document_weights__weight": "weight"
#         # })
#         # base_weights = base_weights.groupby("pk").apply(numpy.prod)
#         # training_data = training_data.merge(base_weights, on="pk", how="left")
#         # training_data['weight'] = training_data['weight'].fillna(1.0)
#
#         for frame in self.frames.all():
#             code_weights = frame.get_params().get("code_weights", [])
#             if len(code_weights) > 0:
#
#                 weight_var_names, weight_var_functions = zip(*code_weights)
#
#                 print "Adding additional frame weights: {}".format(weight_var_names)
#
#                 frame = self.filter_documents(frame.documents.all(), additional_columns=weight_var_names)
#                 valid_partition = False
#                 for i, var in enumerate(weight_var_names):
#                     frame[var] = frame[var].map(weight_var_functions[i])
#                     if len(frame[var].value_counts()) > 1:
#                         valid_partition = True
#
#                 if valid_partition:
#                     weight_vars = []
#                     for var in weight_var_names:
#                         var_frame = frame.dropna(subset=[var])[["pk", var]]
#                         dummies = pandas.get_dummies(var_frame, prefix=var, columns=[var])
#                         weight_vars.extend([d for d in dummies.columns if d.startswith(var)])
#                         frame = frame.merge(dummies, on="pk", how="left")
#
#                     training_sample = frame[frame['pk'].isin(training_data['pk'].values)]
#                     training_sample['weight'] = compute_sample_weights_from_frame(frame, training_sample, weight_vars)
#                     training_data['frame_weight'] = \
#                     training_data.merge(training_sample[["pk", "weight"]], on="pk", how="left")["weight"]
#                     training_data['frame_weight'] = training_data['frame_weight'].fillna(1.0)
#                     training_data['training_weight'] = training_data['training_weight'] * training_data['frame_weight']  # remember, training_weight was set initially by sampling_weight
#                     # TODO: for documents that belong to multiple frames, this will NOT work as intended and could multiply some rows into the extremes
#                     del training_data["frame_weight"]
#
#         return training_data
#
#
#     def _add_balancing_weights(self, training_data):
#
#         sample = copy.copy(training_data)
#
#         weight_var_names = []
#         for mapper_name in self.parameters["documents"].get("balancing_weights", []):
#             balancing_module = importlib.import_module(
#                 "logos.learning.utils.balancing_variables.{0}".format(mapper_name))
#             sample[mapper_name] = sample.apply(balancing_module.var_mapper, axis=1)
#             weight_var_names.append(mapper_name)
#
#         if len(self.document_types) > 1 and self.parameters["documents"].get("balance_document_types", False):
#             weight_var_names.append("document_type")
#
#         if len(weight_var_names) > 0:
#
#             print "Applying balancing variables: {}".format(weight_var_names)
#
#             weight_vars = []
#             for var in weight_var_names:
#                 var_sample = sample.dropna(subset=[var])[["pk", var]]
#                 dummies = pandas.get_dummies(var_sample, prefix=var, columns=[var])
#                 weight_vars.extend([d for d in dummies.columns if d.startswith(var)])
#                 sample = sample.merge(dummies, on="pk", how="left")
#
#             sample['weight'] = compute_balanced_sample_weights(sample, weight_vars, weight_column="sampling_weight")
#             training_data['balancing_weight'] = training_data.merge(sample[["pk", "weight"]], on="pk", how="left")[
#                 "weight"]
#             training_data["balancing_weight"] = training_data["balancing_weight"].fillna(1.0)
#             training_data["training_weight"] = training_data["training_weight"] * training_data["balancing_weight"]
#             del training_data["balancing_weight"]
#
#             # print "Balancing document types"
#             #
#             # balanced_ratio = 1.0 / float(len(self.document_types))
#             #
#             # if use_frames:
#             #     frame = self.filter_documents(get_model("Document").objects.filter(sample_frames__in=self.frames.all()), additional_columns=["{}_id".format(d) for d in self.document_types])
#             #     for doc_type in self.document_types:
#             #         weight = balanced_ratio / (float(len(frame[~frame["{}_id".format(doc_type)].isnull()])) / float(len(frame)))
#             #         print "Balancing {} with weight {}, based on frames".format(doc_type, weight)
#             #         training_data.ix[training_data['document_type'] == doc_type, "training_weight"] = training_data["training_weight"] * weight
#             #
#             # else:
#             #     for doc_type in self.document_types:
#             #         weight = balanced_ratio / (float(len(training_data[training_data['document_type'] == doc_type])) / float(len(training_data)))
#             #         print "Balancing {} with weight {}, based on sample".format(doc_type, weight)
#             #         training_data.ix[training_data['document_type'] == doc_type, "training_weight"] = training_data["training_weight"] * weight
#
#         return training_data
#
#
#     @require_model
#     def show_top_features(self, n=10):
#
#         print "Top features: "
#
#         if hasattr(self.model.best_estimator_, "named_steps"): steps = self.model.best_estimator_.named_steps
#         else: steps = self.model.best_estimator_.steps
#
#         feature_names = self.get_feature_names(self.model.best_estimator_)
#         class_labels = steps['model'].classes_
#
#         top_features = {}
#         if hasattr(steps['model'], "coef_"):
#             if len(class_labels) == 2:
#                 try: coefs = steps['model'].coef_.toarray()[0]
#                 except: coefs = steps['model'].coef_[0]
#                 values = list(self.code_variable.codes.values_list("value", flat=True).order_by("pk"))
#                 top_features["{0} ({1})".format(self.code_variable.codes.get(value=values[0]).label, values[0])] = sorted(zip(coefs, feature_names))[:n]
#                 top_features["{0} ({1})".format(self.code_variable.codes.get(value=values[1]).label, values[1])] = sorted(zip(coefs, feature_names))[:-(n+1):-1]
#             else:
#                 for i, class_label in enumerate(class_labels):
#                     try: coefs = steps['model'].coef_.toarray()[i]
#                     except: coefs = steps['model'].coef_[i]
#                     top_features["{0} ({1})".format(self.code_variable.codes.get(pk=class_label).label, class_label)] = sorted(zip(coefs, feature_names))[-n:]
#         elif hasattr(steps['model'], "feature_importances_"):
#             top_features["n/a"] = sorted(zip(
#                 steps['model'].feature_importances_,
#                 feature_names
#             ))[:-(n+1):-1]
#
#         for class_label, top_n in top_features.iteritems():
#             print class_label
#             for c, f in top_n:
#                 try: print "\t%.4f\t\t%-15s" % (c, f)
#                 except:
#                     print "Error: {}, {}".format(c, f)
#
#
#     @require_training_data
#     @require_model
#     def print_report(self):
#
#         super(DocumentClassificationRegressionHandler, self).print_report()
#
#
#     @require_training_data
#     def _get_model(self, pipeline_steps, params):
#
#         results = super(DocumentClassificationRegressionHandler, self)._get_model(pipeline_steps, params)
#
#         if not results['predict_y']:
#             print "No test data was provided, scanning database for validation data instead"
#             validation_df = self._get_training_data(validation=True)['training_data']
#             if is_not_null(validation_df):
#                 print "Validation data found, computing predictions"
#                 results['test_y'] = validation_df[self.outcome_variable]
#                 X_cols = validation_df.columns.tolist()
#                 X_cols.remove(self.outcome_variable)
#                 results['test_x'] = validation_df[X_cols]
#                 results['test_ids'] = results['test_y'].index
#                 results['predict_y'] = results['model'].predict(results['test_x'])
#
#         def mapper(x):
#             if x == max(results['test_y']): return 1
#             elif x == min(results['test_y']): return 0
#             else: return None
#
#         test_y_binary = results['test_y'].map(mapper)
#         comparison = [z for z in zip(test_y_binary, results['predict_y'], results['test_x']['sampling_weight']) if is_not_null(z[0])]
#         test_y_binary = [z[0] for z in comparison]
#         predict_y = [z[1] for z in comparison]
#         weights = [z[2] for z in comparison]
#
#         scorer = self._get_scoring_function(self.parameters["model"]["binary_scoring_function"], binary_base_code=0)._score_func
#         results['predict_threshold'] = self._get_best_binary_threshold(predict_y, test_y_binary, scorer, weights=weights)
#         predict_y_binary = pandas.Series(results['predict_y']).map(lambda x: 1 if x >= results['predict_threshold'] else 0)
#         results['training_threshold'] = self._get_best_binary_threshold(results['test_y'], predict_y_binary, scorer, weights=results['test_x']['sampling_weight'])
#
#         return results
#
#     def _get_thresholds_simultaneously(self, vals, benchmarks, score_func, weights=None):
#         best_score = (0, 0, 0)
#         for v_threshold in numpy.linspace(min(vals), max(vals), 100):
#             val_binary = [1 if v >= v_threshold else 0 for v in vals]
#             for b_threshold in numpy.linspace(min(benchmarks), max(benchmarks), 100):
#                 benchmark_binary = [1 if b >= b_threshold else 0 for b in benchmarks]
#                 score = score_func(benchmark_binary, val_binary, sample_weight=weights)
#                 if not numpy.isinf(score) and score > 0 and score > best_score[2] and v_threshold >= .2 and v_threshold <= .8:
#                     best_score = (v_threshold, b_threshold, score)
#         print best_score[2]
#         return (best_score[0], best_score[1])
#
#     def _get_best_binary_threshold(self, vals, benchmarks, score_func, weights=None):
#         best_score = (0, 0)
#         for threshold in numpy.linspace(min(vals), max(vals), 100):
#             val_binary = [1 if v >= threshold else 0 for v in vals]
#             score = score_func(benchmarks, val_binary, sample_weight=weights)
#             if score > best_score[1]:
#                 best_score = (threshold, score)
#         return best_score[0]
#
#     def _convert_training_array_to_classes(self, vals, threshold_override=None):
#         t = self.training_threshold if not threshold_override else threshold_override
#         code_map = {c["value"]: c["pk"] for c in self.code_variable.codes.values("pk", "value")}
#         try: return vals.map(lambda x: code_map["1"] if x >= t else code_map["0"])
#         except: return pandas.Series(vals).map(lambda x: code_map["1"] if x >= t else code_map["0"])
#
#     def _convert_predicted_array_to_classes(self, vals, threshold_override=None):
#         t = self.predict_threshold if not threshold_override else threshold_override
#         code_map = {c["value"]: c["pk"] for c in self.code_variable.codes.values("pk", "value")}
#         try: return vals.map(lambda x: code_map["1"] if x >= t else code_map["0"])
#         except: return pandas.Series(vals).map(lambda x: code_map["1"] if x >= t else code_map["0"])
#
#
#     @require_training_data
#     @require_model
#     def get_report_results(self):
#
#         test_y = self._convert_training_array_to_classes(self.test_y)
#         predict_y = self._convert_predicted_array_to_classes(self.predict_y)
#         rows = []
#         report = classification_report(test_y, predict_y, sample_weight=self.test_x['sampling_weight'] if self.parameters["model"]["use_sample_weights"] else None)
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
#
#     @require_model
#     def save_to_database(self, name, compute_cv_scores=False):
#
#         if not self.saved_model:
#
#             params = copy.copy(self.parameters)
#             # if "frames" in self.parameters["documents"]:
#             #     params["documents"]["frames"] = [f['name'] for f in self.parameters["documents"]["frames"]]
#
#             print "Saving model '{}' to database".format(name)
#
#             frames = self.frames.all()
#             self.saved_model = get_model("DocumentClassificationModel").objects.create_or_update(
#                 {"name": name},
#                 {
#                     "variable": self.code_variable,
#                     "document_types": self.document_types,
#                     "pipeline_name": self.pipeline_name,
#                     "parameters": params
#                 },
#                 save_nulls=True,
#                 empty_lists_are_null=True
#             )
#             self.saved_model.frames = frames
#             self.saved_model.save()
#             if compute_cv_scores:
#                 print "Computing CV scores on training set"
#                 self.saved_model.compute_cv_scores(num_folds=5)
#                 # print "Computing CV scores on test set"
#                 # self.saved_model.compute_cv_scores(use_test_data=True, num_folds=10)
#
#         else:
#
#             print "Classifier already saved to database as '{}'!".format(self.saved_model.name)
#
#
#     @require_model
#     def apply_model(self, documents, keep_cols=None):
#
#         if not keep_cols: keep_cols = ["pk"]
#         documents = self.filter_documents(documents)
#         predictions = super(DocumentClassificationRegressionHandler, self).apply_model(documents, keep_cols=keep_cols)
#         return self._convert_predicted_array_to_classes(predictions)
#
#
#     @require_model
#     @temp_cache_wrapper
#     def apply_model_to_database(self, documents, num_cores=2, chunk_size=1000, clear_temp_cache=True):
#
#         if not self.saved_model:
#
#             print "Model is not currently saved in the database; please call 'save_to_database(name)' and pass a unique name"
#
#         else:
#
#             try: document_ids = list(documents.values_list("pk", flat=True))
#             except: document_ids = [getattr(d, "pk") for d in documents]
#
#             print "Processing {} {}".format(len(document_ids), self.document_types)
#             for i, chunk in enumerate(chunker(document_ids, chunk_size)):
#                 codes = self.apply_model(get_model("Document").objects.filter(pk__in=chunk))
#                 print "Processing {} of {} ({}, {})".format((i+1)*chunk_size, len(document_ids), self.code_variable.name, self.document_types)
#                 for index, row in codes.iterrows():
#                     get_model("ClassifierDocumentCode").objects.create_or_update(
#                         {
#                             "classifier": self.saved_model,
#                             "document_id": row["pk"]
#                         },
#                         {"code_id": row[self.outcome_variable]},
#                         save_nulls=True,
#                         return_object=False
#                     )
#
#             # pool = Pool(processes=num_cores)
#             # for i, chunk in enumerate(chunker(document_ids, chunk_size)):
#             #     print "Creating chunk %i of %i" % (i+1, (i+1)*chunk_size)
#             #     pool.apply_async(_process_document_chunk, args=(self.saved_model.pk, chunk, i))
#             #     # _process_document_chunk(self.saved_model.pk, chunk, i)
#             #     # break
#             # pool.close()
#             # pool.join()
#
#
#     @require_model
#     def coded_documents(self):
#
#         if self.saved_model:
#             return self.saved_model.coded_documents.all()
#         else:
#             return []
#
#     # TODO: moved this over from the DocumentClassificationModel model, need to iron out and get working
#     # @require_model
#     # def get_code_cv_training_scores(self, fold_preds, code_value="1", partition_by=None, restrict_document_type=None,
#     #                                 min_support=0):
#     #
#     #     # BELOW: code in progress, for converting regression measures to classification scores
#     #     # if hasattr(h, "_convert_training_array_to_classes"):
#     #     #     y = h._convert_training_array_to_classes(y)
#     #     #
#     #     # if hasattr(h, "_convert_predicted_array_to_classes"):
#     #     #     fold_preds = [[h._convert_predicted_array_to_classes(f[0]), f[1]] for f in fold_preds]
#     #
#     #     # TODO: seems like using thresholds based off of CV averages/medians is much more accurate
#     #     # So we need some way of folding this back into the DocumentClassificationRegressionHandler
#     #     # Maybe we just have apply_model take threshold parameters, which can be based from the DocumentClassificationModel?
#     #     # if hasattr(h, "_get_thresholds_simultaneously"):
#     #     #     y_thresholds = []
#     #     #     pred_thresholds = []
#     #     #
#     #     #     score_func = h._get_scoring_function("maxmin", binary_base_code=0)._score_func
#     #     #     best_score = (0, 0, 0)
#     #     #     for y_threshold in numpy.linspace(min(y), max(y), 50):
#     #     #         if y_threshold > min(y):
#     #     #             print y_threshold
#     #     #             for p_threshold in numpy.linspace(min(fold_preds[0][0]), max(fold_preds[0][0]), 50):
#     #     #                 if p_threshold > min(fold_preds[0][0]):
#     #     #                     scores = []
#     #     #                     for preds, indices in fold_preds:
#     #     #                         y_binary = pandas.Series([1 if v >= y_threshold else 0 for v in y.iloc[indices]])
#     #     #                         pred_binary = pandas.Series([1 if b >= p_threshold else 0 for b in preds])
#     #     #                         score = score_func(y_binary, pred_binary, sample_weight=X['sampling_weight'][indices])
#     #     #                         scores.append(score)
#     #     #                     score = numpy.average(scores)
#     #     #                     #print "{}, {}: {}".format(y_threshold, p_threshold, score)
#     #     #                     if score > best_score[2]:
#     #     #                         best_score = (y_threshold, p_threshold, score)
#     #     #             print best_score
#     #     #     print best_score
#     #     #
#     #     #     # for preds, indices in fold_preds:
#     #     #     #
#     #     #     #     y_threshold, pred_threshold = h._get_thresholds_simultaneously(
#     #     #     #         y.iloc[indices],
#     #     #     #         preds,
#     #     #     #         #h._get_scoring_function(h.parameters["model"]["binary_scoring_function"],
#     #     #     #         h._get_scoring_function("mean_difference",
#     #     #     #                                 binary_base_code=0)._score_func
#     #     #     #     )
#     #     #     #     y_thresholds.append(y_threshold)
#     #     #     #     pred_thresholds.append(pred_threshold)
#     #     #     #     print "{}, {}".format(y_threshold, pred_threshold)
#     #     #
#     #     #     y = h._convert_training_array_to_classes(y, threshold_override=best_score[0]) #numpy.median(y_thresholds))
#     #     #     fold_preds = [
#     #     #         [h._convert_predicted_array_to_classes(f[0], threshold_override=best_score[1]), #numpy.median(pred_thresholds)),
#     #     #          f[1]] for f in fold_preds]
#     #
#     #     code = self.variable.codes.get(value=code_value).pk
#     #
#     #     scores = {}
#     #     full_index = X.index
#     #     if restrict_document_type:
#     #         full_index = X[X['document_type'] == restrict_document_type].index
#     #
#     #     if len(full_index) > 0:
#     #         if partition_by:
#     #             X['partition'] = X.apply(
#     #                 lambda x: Document.objects.filter(pk=x['document_id']).values_list(partition_by, flat=True)[0],
#     #                 axis=1
#     #             )
#     #             for partition in X['partition'].unique():
#     #                 index = full_index.intersection(X[X['partition'] == partition].index)
#     #                 rows = []
#     #                 for fold, preds in enumerate(fold_preds):
#     #                     preds, indices = preds
#     #                     new_indices = list(set(indices).intersection(set(index.values)))
#     #                     preds = preds[map(map(lambda x: (x), indices).index, new_indices)]
#     #                     indices = new_indices
#     #                     if len(indices) > min_support:
#     #                         weights = X['sampling_weight'][indices]
#     #                         code_preds = [1 if x == code else 0 for x in preds]
#     #                         code_true = [1 if x == code else 0 for x in y[indices]]
#     #                         if sum(code_preds) > 0 and sum(code_true) > 0:
#     #                             rows.append(self._get_scores(code_preds, code_true, weights=weights))
#     #                 if len(rows) > 0:
#     #                     means = {"{}_mean".format(k): v for k, v in pandas.DataFrame(rows).mean().to_dict().iteritems()}
#     #                     stds = {"{}_std".format(k): v for k, v in pandas.DataFrame(rows).std().to_dict().iteritems()}
#     #                     errs = {"{}_err".format(k): v for k, v in pandas.DataFrame(rows).sem().to_dict().iteritems()}
#     #                     scores[partition] = {}
#     #                     scores[partition].update(means)
#     #                     scores[partition].update(stds)
#     #                     scores[partition].update(errs)
#     #                     if len(scores[partition].keys()) == 0:
#     #                         del scores[partition]
#     #
#     #         else:
#     #             rows = []
#     #             for fold, preds in enumerate(fold_preds):
#     #                 preds, indices = preds
#     #                 new_indices = list(set(indices).intersection(set(full_index.values)))
#     #                 preds = preds[map(map(lambda x: (x), indices).index, new_indices)]
#     #                 indices = new_indices
#     #                 if len(indices) > min_support:
#     #                     weights = X['sampling_weight'][indices]
#     #                     code_preds = [1 if x == code else 0 for x in preds]
#     #                     code_true = [1 if x == code else 0 for x in y[indices]]
#     #                     if sum(code_preds) > 0 and sum(code_true) > 0:
#     #                         rows.append(self._get_scores(code_preds, code_true, weights=weights))
#     #             if len(rows) > 0:
#     #                 means = {"{}_mean".format(k): v for k, v in pandas.DataFrame(rows).mean().to_dict().iteritems()}
#     #                 stds = {"{}_std".format(k): v for k, v in pandas.DataFrame(rows).std().to_dict().iteritems()}
#     #                 errs = {"{}_err".format(k): v for k, v in pandas.DataFrame(rows).sem().to_dict().iteritems()}
#     #                 scores.update(means)
#     #                 scores.update(stds)
#     #                 scores.update(errs)
#     #
#     #     if len(scores.keys()) > 0:
#     #         return scores
#     #     else:
#     #         return None
#
#     # @require_model
#     # def get_code_validation_test_scores(self, code_value="1", partition_by=None, restrict_document_type=None,
#     #                                     use_expert_consensus_subset=False, compute_for_experts=False, min_support=0):
#     #
#     #     X = h.test_x
#     #     test_y = h.test_y
#     #     predict_y = pandas.DataFrame(h.predict_y, columns=["predict_y"], index=X.index)["predict_y"]
#     #
#     #     # if hasattr(h, "_convert_training_array_to_classes"):
#     #     #     test_y = h._convert_training_array_to_classes(test_y)
#     #     #
#     #     # if hasattr(h, "_convert_predicted_array_to_classes"):
#     #     #     predict_y = h._convert_predicted_array_to_classes(predict_y)
#     #
#     #     code = self.variable.codes.get(value=code_value).pk
#     #
#     #     if restrict_document_type:
#     #         X = X[X['document_type'] == restrict_document_type]
#     #
#     #     abort = False
#     #     if use_expert_consensus_subset:
#     #         expert_codes = get_model("CoderDocumentCode").objects \
#     #             .filter(sample_unit__sample__frame__in=self.frames.all()) \
#     #             .filter(coder__is_mturk=False) \
#     #             .filter(code__variable=self.variable) \
#     #             .filter(coder__name__in=['ahughes', 'pvankessel']) \
#     #             .filter(sample_unit__document_id__in=X['pk'].values) \
#     #             .values("sample_unit__document_id", "code__value", "code_id")
#     #         expert_df = pandas.DataFrame.from_records(expert_codes)
#     #         if len(expert_df) > 0:
#     #             expert_df['code'] = expert_df['code_id'].map(lambda x: 1 if int(x) == int(code) else 0)
#     #             expert_df_mean = expert_df.groupby("sample_unit__document_id").mean().reset_index()
#     #             expert_df_mean = X[["pk"]].merge(expert_df_mean, left_on="pk", right_on="sample_unit__document_id",
#     #                                              how="left")
#     #             expert_df_mean.index = X.index
#     #             expert_df_mean = expert_df_mean.dropna(subset=["code_id"])
#     #             consensus_ids = expert_df_mean[expert_df_mean['code'].isin([0.0, 1.0])].index
#     #             X = X.ix[consensus_ids]  # X[X['pk'].isin(consensus_ids)]
#     #             if compute_for_experts:
#     #                 test_y = expert_df_mean['code_id'].map(lambda x: int(x))
#     #         else:
#     #             abort = True
#     #
#     #     if not abort:
#     #
#     #         test_y = test_y[X.index]
#     #         predict_y = predict_y[X.index]
#     #
#     #         scores = {}
#     #         if len(X) > 0:
#     #             if partition_by:
#     #                 scores = {}
#     #                 X['partition'] = X.apply(
#     #                     lambda x: Document.objects.filter(pk=x['document_id']).values_list(partition_by, flat=True)[0],
#     #                     axis=1)
#     #                 for partition in X['partition'].unique():
#     #                     index = X[X['partition'] == partition].index
#     #                     if len(index) > min_support:
#     #                         weights = X['sampling_weight'][index]
#     #                         code_preds = [1 if x == code else 0 for x in predict_y[index]]
#     #                         code_true = [1 if x == code else 0 for x in test_y[index]]
#     #                         if sum(code_preds) > 0 and sum(code_true) > 0:
#     #                             scores[partition] = self._get_scores(code_preds, code_true, weights=weights)
#     #                             if len(scores[partition].keys()) == 0:
#     #                                 del scores[partition]
#     #
#     #             elif len(X) > min_support:
#     #                 weights = X['sampling_weight']
#     #                 code_preds = [1 if x == code else 0 for x in predict_y]
#     #                 code_true = [1 if x == code else 0 for x in test_y]
#     #                 if sum(code_preds) > 0 and sum(code_true) > 0:
#     #                     scores = self._get_scores(code_preds, code_true, weights=weights)
#     #
#     #         if len(scores.keys()) > 0:
#     #             return scores
#     #         else:
#     #             return None
#     #
#     #     else:
#     #
#     #         return None
#
# def _process_document_chunk(model_id, chunk, i):
#
#     try:
#
#         import os, django, sys, traceback
#         os.environ.setdefault("DJANGO_SETTINGS_MODULE", "logos.settings")
#         django.setup()
#         from django.db import connection
#         connection.close()
#
#         from logos.models import CodeVariableClassifier, ClassifierDocumentCode
#
#         model = CodeVariableClassifier.objects.get(pk=model_id)
#         ClassifierDocumentCode.objects.filter(document_id__in=chunk, classifier=model)
#         new_codes = model.handler.code_documents(get_model("Document").objects.filter(pk__in=chunk))
#
#         doc_codes = []
#         for code in new_codes:
#             code.update({"classifier_id": model_id})
#             doc_codes.append(
#                 ClassifierDocumentCode(**code)
#             )
#         ClassifierDocumentCode.objects.bulk_create(doc_codes)
#
#         print "Done processing chunk %i" % (int(i)+1)
#
#     except Exception as e:
#
#         print e
#         exc_type, exc_value, exc_traceback = sys.exc_info()
#         print exc_type
#         print exc_value
#         print exc_traceback
#         traceback.print_exc(exc_traceback)
#         raise