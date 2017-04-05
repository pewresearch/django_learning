import importlib, copy, pandas, numpy

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

from sklearn.cross_validation import train_test_split, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss, make_scorer, mean_squared_error, r2_score, matthews_corrcoef, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind

from django_commander.models import LoggedExtendedModel

from django_learning.utils import get_document_types, get_pipeline_repr, get_param_repr
from django_learning.utils.pipelines import pipelines
from django_learning.utils.dataset_extractors import dataset_extractors
from django_learning.utils.decorators import require_training_data, require_model, temp_cache_wrapper
from django_learning.utils.feature_extractors import BasicExtractor
from django_learning.utils.models import models as learning_models
from django_learning.models.learning import LearningModel, DocumentLearningModel

from pewtils import is_not_null, is_null, decode_text, recursive_update
from pewtils.django import get_model, CacheHandler
from pewtils.sampling import compute_sample_weights_from_frame, compute_balanced_sample_weights
from pewtils.stats import wmom


class Classification(LoggedExtendedModel):

    document = models.ForeignKey("django_learning.Document", related_name="classifications")
    label = models.ForeignKey("django_learning.Label", related_name="classifications")

    # classifier = models.ForeignKey("django_learning.Classifier", related_name="classifications")
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    model = GenericForeignKey('content_type', 'object_id')

    probability = models.FloatField(null=True, help_text="The probability of the assigned label, if applicable")

    # def validate_unique(self, *args, **kwargs):
    #        super(Classification, self).validate_unique(*args, **kwargs)
    #        if not self.id:
    #            if not self.label.question.multiple:
    #                if self.model.objects\
    #                        .filter(label__question=self.label.question)\
    #                        .filter(document=self.document)\
    #                        .exists():
    #                    raise ValidationError(
    #                        {
    #                            NON_FIELD_ERRORS: [
    #                                'Classification with the same variable already exists'
    #                            ],
    #                        }
    #                    )

    def __repr__(self):
       return "<Classification label={0}, document={2}>".format(
           self.label, self.document
       )


class ClassificationModel(LearningModel):

    classifications = GenericRelation(Classification)

    @require_model
    def show_top_features(self, n=10):

        if hasattr(self.model.best_estimator_, "named_steps"):
            steps = self.model.best_estimator_.named_steps
        else:
            steps = self.model.best_estimator_.steps

        feature_names = self.get_feature_names(self.model.best_estimator_)
        class_labels = steps['model'].classes_

        top_features = {}
        if hasattr(steps['model'], "coef_"):
            if len(class_labels) == 2:
                top_features[0] = sorted(zip(
                    steps['model'].coef_[0],
                    feature_names
                ))[:n]
                top_features[1] = sorted(zip(
                    steps['model'].coef_[0],
                    feature_names
                ))[:-(n + 1):-1]
            else:
                for i, class_label in enumerate(class_labels):
                    top_features[class_label] = sorted(zip(
                        steps['model'].coef_[i],
                        feature_names
                    ))[-n:]
        elif hasattr(steps['model'], "feature_importances_"):
            top_features["n/a"] = sorted(zip(
                steps['model'].feature_importances_,
                feature_names
            ))[:-(n + 1):-1]

        for class_label, top_n in top_features.iteritems():
            print class_label
            for c, f in top_n:
                try:
                    print "\t%.4f\t\t%-15s" % (c, f)
                except:
                    print "Error: {}, {}".format(c, f)

    @require_model
    def print_report(self):

        super(ClassificationModel, self).print_report()

        print "Detailed classification report:"
        print classification_report(self.test_y, self.predict_y,
                                    sample_weight=self.test_x['sampling_weight'] if self.parameters["model"].get(
                                        "use_sample_weights", False) else None)

        print "Confusion matrix:"
        print confusion_matrix(self.test_y, self.predict_y)

        self.show_top_features()

    @require_model
    def get_incorrect_predictions(self, actual_code=None):

        df = pandas.concat([self.test_y, self.test_x], axis=1)
        df['prediction'] = self.predict_y
        df = df[df[self.outcome_column] != df['prediction']]
        if actual_code:
            df = df[df[self.outcome_column] == actual_code]
        return df

    @require_model
    def get_report_results(self):

        rows = []
        report = classification_report(self.test_y, self.predict_y,
                                       sample_weight=self.test_x['sampling_weight'] if self.parameters["model"].get(
                                           "use_sample_weights", False) else None)
        for row in report.split("\n"):
            row = row.strip().split()
            if len(row) == 7:
                row = row[2:]
            if len(row) == 5:
                rows.append({
                    "class": row[0],
                    "precision": row[1],
                    "recall": row[2],
                    "f1-score": row[3],
                    "support": row[4]
                })
        return rows


class DocumentClassificationModel(ClassificationModel, DocumentLearningModel):

    def extract_training_dataset(self, refresh=False):

        super(DocumentClassificationModel, self).extract_training_dataset(refresh=refresh)
        self.training_dataset["training_weight"] = self.training_dataset["sampling_weight"]
        for additional_weight in ["balancing_weight"]:
            if additional_weight in self.training_dataset.columns:
                print "Mixing {} into the training weights".format(additional_weight)
                self.training_dataset["training_weight"] = self.training_dataset["training_weight"] * self.training_dataset[additional_weight]
        self.document_types = self.training_dataset["document_type"].unique()

    def _get_all_document_types(self):

        return [f.name for f in get_model("Document", app_name="django_learning").get_parent_relations()]

    # @require_training_data
    # def _train_model(self, pipeline_steps, params, **kwargs):
    #
    #     results = super(DocumentClassificationModel, self)._train_model(pipeline_steps, params, **kwargs)
    #
    #     if not results['predict_y']:
    #         print "No hold-out test data was set aside, scanning database for validation data instead"
    #         validation_df = self._get_training_data(sample_type="validation")['training_data']
    #         if is_not_null(validation_df):
    #             print "Validation data found, computing predictions"
    #             results['test_y'] = validation_df[self.outcome_column]
    #             X_cols = validation_df.columns.tolist()
    #             X_cols.remove(self.outcome_column)
    #             results['test_x'] = validation_df[X_cols]
    #             results['test_ids'] = results['test_y'].index
    #             results['predict_y'] = results['model'].predict(results['test_x'])
    #
    #     return results
    #
    # @require_training_data
    # @require_model
    # def get_report_results(self):
    #
    #     rows = []
    #     # report = classification_report(self.test_y, self.predict_y, sample_weight=self.test_x['sampling_weight'] if self.parameters["model"]["use_sample_weights"] else None)
    #
    #     predict_y = pandas.Series(self.predict_y, index=self.test_x.index)
    #     test_x = self.test_x[~self.test_x['document_id'].isin(self.train_x['document_id'])]
    #     test_y = self.test_y.ix[test_x.index]
    #     predict_y = predict_y[test_x.index].values
    #
    #     report = classification_report(test_y, predict_y, sample_weight=test_x['sampling_weight'])
    #     for row in report.split("\n"):
    #         row = row.strip().split()
    #         if len(row) == 7:
    #             row = row[2:]
    #         if len(row) == 5:
    #             rows.append({
    #                 "class": row[0],
    #                 "precision": row[1],
    #                 "recall": row[2],
    #                 "f1-score": row[3],
    #                 "support": row[4]
    #             })
    #     return rows
    #
    #
    # @require_model
    # @temp_cache_wrapper
    # def compute_cv_folds(self, use_test_data=False, num_folds=5, clear_temp_cache=True, refresh=False):
    #
    #     fold_preds = None
    #     if (not use_test_data and (is_null(self.cv_folds) or refresh)) or \
    #             (use_test_data and (is_null(self.cv_folds_test) or refresh)):
    #
    #         if use_test_data:
    #             train_x = self.test_x[~self.test_x['document_id'].isin(self.train_x['document_id'])]
    #             train_y = self.test_y.ix[train_x.index]
    #         else:
    #             train_x = self.train_x
    #             train_y = self.train_y
    #
    #         final_model = self.model.best_estimator_
    #         fold_preds = cross_val_predict(final_model, train_x, train_y,
    #                                        keep_folds_separate=True,
    #                                        cv=num_folds,
    #                                        fit_params={
    #                                            'model__sample_weight': [x for x in train_x["sampling_weight"].values]
    #                                        }
    #                                        )
    #         # TODO: you might be able to get around your custom sklearn hack
    #         # see: https://stackoverflow.com/questions/42011850/is-there-a-way-to-see-the-folds-for-cross-validation-in-gridsearchcv
    #
    #     if fold_preds:
    #         if use_test_data:
    #             self.cv_folds_test = fold_preds
    #         else:
    #             self.cv_folds = fold_preds
    #         self.save()
    #
    # @require_model
    # def get_code_cv_training_scores(self, use_test_data=False, label_value="1", partition_by=None,
    #                                 restrict_document_type=None, min_support=0):
    #
    #     if use_test_data:
    #         X = self.test_x
    #         y = self.test_y
    #         fold_preds = self.cv_folds_test
    #     else:
    #         X = self.train_x
    #         y = self.train_y
    #         fold_preds = self.cv_folds
    #
    #     label = self.question.labels.get(value=label_value).pk
    #
    #     scores = {}
    #     full_index = X.index
    #     if restrict_document_type and 'document_type' in X.columns:
    #         full_index = X[X['document_type'] == restrict_document_type].index
    #
    #     if len(full_index) > 0:
    #         if partition_by:
    #             X['partition'] = X.apply(
    #                 lambda x:
    #                 get_model("Document").objects.filter(pk=x['document_id']).values_list(partition_by, flat=True)[0],
    #                 axis=1
    #             )
    #             for partition in X['partition'].unique():
    #                 index = full_index.intersection(X[X['partition'] == partition].index)
    #                 rows = []
    #                 for fold, preds in enumerate(fold_preds):
    #                     preds, indices = preds
    #                     new_indices = list(set(indices).intersection(set(index.values)))
    #                     preds = preds[map(map(lambda x: (x), indices).index, new_indices)]
    #                     indices = new_indices
    #                     if len(indices) > min_support:
    #                         weights = X['sampling_weight'][indices]
    #                         label_preds = [1 if x == label else 0 for x in preds]
    #                         label_true = [1 if x == label else 0 for x in y[indices]]
    #                         if sum(label_preds) > 0 and sum(label_true) > 0:
    #                             rows.append(self._get_scores(label_preds, label_true, weights=weights))
    #                 if len(rows) > 0:
    #                     means = {"{}_mean".format(k): v for k, v in pandas.DataFrame(rows).mean().to_dict().iteritems()}
    #                     stds = {"{}_std".format(k): v for k, v in pandas.DataFrame(rows).std().to_dict().iteritems()}
    #                     errs = {"{}_err".format(k): v for k, v in pandas.DataFrame(rows).sem().to_dict().iteritems()}
    #                     scores[partition] = {}
    #                     scores[partition].update(means)
    #                     scores[partition].update(stds)
    #                     scores[partition].update(errs)
    #                     if len(scores[partition].keys()) == 0:
    #                         del scores[partition]
    #
    #         else:
    #             rows = []
    #             for fold, preds in enumerate(fold_preds):
    #                 preds, indices = preds
    #                 new_indices = list(set(indices).intersection(set(full_index.values)))
    #                 preds = preds[map(map(lambda x: (x), indices).index, new_indices)]
    #                 indices = new_indices
    #                 if len(indices) > min_support:
    #                     weights = X['sampling_weight'][indices]
    #                     label_preds = [1 if x == label else 0 for x in preds]
    #                     label_true = [1 if x == label else 0 for x in y[indices]]
    #                     if sum(label_preds) > 0 and sum(label_true) > 0:
    #                         rows.append(self._get_scores(label_preds, label_true, weights=weights))
    #             if len(rows) > 0:
    #                 means = {"{}_mean".format(k): v for k, v in pandas.DataFrame(rows).mean().to_dict().iteritems()}
    #                 stds = {"{}_std".format(k): v for k, v in pandas.DataFrame(rows).std().to_dict().iteritems()}
    #                 errs = {"{}_err".format(k): v for k, v in pandas.DataFrame(rows).sem().to_dict().iteritems()}
    #                 scores.update(means)
    #                 scores.update(stds)
    #                 scores.update(errs)
    #
    #     if len(scores.keys()) > 0:
    #         return scores
    #     else:
    #         return None
    #
    # def _get_expert_consensus_dataframe(self, label_value, coders=None, use_consensus_ignore_flag=True):
    #
    #     if not coders:
    #         coders = ['ahughes', 'pvankessel']
    #
    #     df = self._get_raw_codes(turk=False, training=False, coder_names=coders,
    #                              use_consensus_ignore_flag=use_consensus_ignore_flag)
    #
    #     df = df[df['label_id'].notnull()]
    #     df = self._add_label_metadata(df)
    #
    #     input_documents = self.filter_documents(
    #         get_model("Document").objects.filter(pk__in=df["document_id"])
    #     )
    #
    #     df = df.merge(
    #         input_documents,
    #         how='inner',
    #         left_on='document_id',
    #         right_on="pk"
    #     )
    #
    #     if "training_weight" in df.columns:
    #         del df['training_weight']
    #
    #     label = self.question.labels.get(value=label_value).pk
    #     df['label'] = df['label_id'].map(lambda x: 1 if int(x) == int(label) else 0)
    #     df_mean = df.groupby("document_id").mean().reset_index()
    #     df_mean = df_mean[df_mean['label'].isin([0.0, 1.0])]
    #
    #     return df_mean
    #
    #
    # @require_model
    # def get_code_validation_test_scores(self, label_value="1", partition_by=None, restrict_document_type=None, use_expert_consensus_subset=False, compute_for_experts=False, min_support=0):
    #
    #     self.predict_y = pandas.Series(self.predict_y, index=self.test_x.index)
    #     self.test_x = self.test_x[~self.test_x['document_id'].isin(self.train_x['document_id'])]
    #     self.test_y = self.test_y.ix[self.test_x.index]
    #     self.predict_y = self.predict_y[self.test_x.index].values
    #
    #     X = self.test_x
    #     test_y = self.test_y
    #     predict_y = pandas.Series(self.predict_y, index=X.index)
    #     # predict_y = pandas.DataFrame()
    #     # predict_y['predict_y'] = self.predict_y
    #     # predict_y = pandas.DataFrame(self.predict_y, columns=["predict_y"], index=X.index)["predict_y"]
    #
    #     label = self.question.labels.get(value=label_value).pk
    #
    #     if restrict_document_type:
    #         X = X[X['document_type']==restrict_document_type]
    #
    #     abort = False
    #     if use_expert_consensus_subset:
    #         expert_consensus_df = self._get_expert_consensus_dataframe(label_value)
    #         if len(expert_consensus_df) > 0:
    #             # X = pandas.merge(X, expert_consensus_df, on='document_id')
    #             old_index = X.index
    #             X = X.merge(expert_consensus_df[['label', 'label_id', 'document_id']], how='left', on='document_id')
    #             X.index = old_index
    #             X = X[~X['label'].isnull()]
    #             if compute_for_experts:
    #                 test_y = X['label_id']
    #             del X['label_id']
    #             del X['label']
    #         else:
    #             abort = True
    #
    #     if not abort:
    #
    #         test_y = test_y.ix[X.index].values
    #         predict_y = predict_y.ix[X.index].values
    #
    #         scores = {}
    #         if len(X) > 0:
    #             if partition_by:
    #                 scores = {}
    #                 X['partition'] = X.apply(lambda x: get_model("Document").objects.filter(pk=x['document_id']).values_list(partition_by, flat=True)[0], axis=1)
    #                 for partition in X['partition'].unique():
    #                     index = X[X['partition']==partition].index
    #                     if len(index) > min_support:
    #                         weights = X['sampling_weight'][index]
    #                         label_preds = [1 if x == label else 0 for x in predict_y[index]]
    #                         label_true = [1 if x == label else 0 for x in test_y[index]]
    #                         if sum(label_preds) > 0 and sum(label_true) > 0:
    #                             scores[partition] = self._get_scores(label_preds, label_true, weights=weights)
    #                             if len(scores[partition].keys()) == 0:
    #                                 del scores[partition]
    #
    #             elif len(X) > min_support:
    #                 weights = X['sampling_weight']
    #                 label_preds = [1 if x == label else 0 for x in predict_y]
    #                 label_true = [1 if x == label else 0 for x in test_y]
    #                 if sum(label_preds) > 0 and sum(label_true) > 0:
    #                     scores = self._get_scores(label_preds, label_true, weights=weights)
    #
    #         if len(scores.keys()) > 0:
    #             return scores
    #         else:
    #             return None
    #
    #     else:
    #
    #         return None
    #
    # def _get_scores(self, label_preds, label_true, weights=None):
    #
    #     row = {
    #         "matthews_corrcoef": matthews_corrcoef(label_true, label_preds, sample_weight=weights),
    #         "accuracy": accuracy_score(label_true, label_preds, sample_weight=weights),
    #         "f1": f1_score(label_true, label_preds, pos_label=1, sample_weight=weights),
    #         "precision": precision_score(label_true, label_preds, pos_label=1, sample_weight=weights),
    #         "recall": recall_score(label_true, label_preds, pos_label=1, sample_weight=weights),
    #         "roc_auc": roc_auc_score(label_true, label_preds, sample_weight=weights) if len(
    #             numpy.unique(label_preds)) > 1 and len(numpy.unique(label_true)) > 1 else None
    #     }
    #
    #     for labelsetname, labelset in [
    #         ("pred", label_preds),
    #         ("true", label_true)
    #     ]:
    #         # unweighted = wmom(codeset, [1.0 for x in codeset], calcerr=True, sdev=True)
    #         weighted = wmom(labelset, weights, calcerr=True, sdev=True)
    #         # for valname, val in zip(["mean", "err", "std"], list(unweighted)):
    #         #     row["{}_{}".format(codesetname, valname)] = val
    #         for valname, val in zip(["mean", "err", "std"], list(weighted)):
    #             row["{}_{}".format(labelsetname, valname)] = val
    #
    #     row["ttest_t"], row["ttest_p"] = ttest_ind(label_preds, label_true)
    #     if row["ttest_p"] > .05:
    #         row["ttest_pass"] = 1
    #     else:
    #         row["ttest_pass"] = 0
    #
    #     row["pct_agree"] = numpy.average([1 if c[0] == c[1] else 0 for c in zip(label_preds, label_true)])
    #
    #     if sum(label_preds) > 0 and sum(label_true) > 0:
    #
    #         result_dict = {0: defaultdict(int), 1: defaultdict(int)}
    #         for pred, true in zip(label_preds, label_true):
    #             result_dict[pred][true] += 1
    #         kappa = cohens_kappa([
    #             [result_dict[0][0], result_dict[0][1]],
    #             [result_dict[1][0], result_dict[1][1]]
    #         ])
    #         row["kappa"] = kappa["kappa"]
    #         row["kappa_err"] = kappa["std_kappa"]
    #
    #     return row
    #
    # @require_model
    # def apply_model(self, documents, keep_cols=None, clear_temp_cache=True):
    #
    #     if not keep_cols: keep_cols = ["pk"]
    #     documents = self.filter_documents(documents)
    #     if len(documents) > 0:
    #         return super(CodeClassificationModel, self).apply_model(documents, keep_cols=keep_cols, clear_temp_cache=clear_temp_cache)
    #     else:
    #         return pandas.DataFrame()

    # @require_model
    # def apply_model_to_frames(self, num_cores=2, chunk_size=1000, refresh_existing=False):
    #
    #     docs = self.frame.documents.all()
    #     if not refresh_existing:
    #         existing = self.coded_documents.values_list("document_id", flat=True)
    #         keep = get_model("Document").objects.filter(
    #             pk__in=set(docs.values_list("pk", flat=True)).difference(set(existing)))
    #         print "Skipping {} existing documents, {} remaining".format(existing.count(), keep.count())
    #         # if existing.count() > 0:
    #         #    docs = docs.exclude(pk__in=existing)
    #         docs = keep
    #     print "Applying model to {} documents".format(docs.count())
    #     self.apply_model_to_database(docs, chunk_size=chunk_size, num_cores=num_cores)

    # @require_model
    # @temp_cache_wrapper
    # def apply_model_to_database(self, documents, num_cores=2, chunk_size=1000, clear_temp_cache=True):
    #
    #     # documents = self._apply_document_filters(documents)
    #
    #     # TODO: edited this to assume that it's a queryset, because it's MUCH more efficient to do a .count check
    #     # if you want to allow for lists of documents, do it in a way that checks whether it's a queryset BEFORE checking length
    #     # so you can avoid having to load/evaluate an unloaded queryset in full, just to check the length
    #     # if len(documents) > 0:
    #     if documents.count() > 0:
    #
    #         try:
    #             document_ids = list(documents.values_list("pk", flat=True))
    #         except:
    #             document_ids = [getattr(d, "pk") for d in documents]
    #
    #         print "Processing {} {}".format(len(document_ids), self.document_types)
    #         # for i, chunk in enumerate(chunker(document_ids, chunk_size)):
    #         #     codes = self.apply_model(get_model("Document").objects.filter(pk__in=chunk))
    #         #     print "Processing {} of {} ({}, {})".format((i+1)*chunk_size, len(document_ids), self.code_variable.name, self.document_types)
    #         #     for index, row in codes.iterrows():
    #         #         get_model("ClassifierDocumentCode").objects.create_or_update(
    #         #             {
    #         #                 "classifier": self.saved_model,
    #         #                 "document_id": row["pk"]
    #         #             },
    #         #             {"code_id": row[self.outcome_variable]},
    #         #             save_nulls=True,
    #         #             return_object=False
    #         #         )
    #
    #         pool = Pool(processes=num_cores)
    #         for i, chunk in enumerate(chunker(document_ids, chunk_size)):
    #             print "Creating chunk %i of %i" % (i + 1, (i + 1) * chunk_size)
    #             pool.apply_async(_process_document_chunk, args=(self.saved_model.pk, chunk, i))
    #             # _process_document_chunk(self.saved_model.pk, chunk, i)
    #             # break
    #         pool.close()
    #         pool.join()
    #
    #     else:
    #
    #         print "All documents were filtered, nothing to do"



# def _process_document_chunk(model_id, chunk, i):
#
#     try:
#
#         import os, django, sys, traceback
#         os.environ.setdefault("DJANGO_SETTINGS_MODULE", "{}.settings".format(settings.SITE_NAME))
#         django.setup()
#         from django.db import connection
#         connection.close()
#
#         from django_learning.models import DocumentClassificationModel, ClassifierDocumentCode
#
#         model = DocumentClassificationModel.objects.get(pk=model_id)
#         ClassifierDocumentCode.objects.filter(document_id__in=chunk, classifier=model).delete()
#         handler = model.handler
#         handler.load_model()
#         codes = handler.apply_model(get_model("Document").objects.filter(pk__in=chunk))
#
#         doc_codes = []
#         for index, row in codes.iterrows():
#             doc_codes.append(
#                 ClassifierDocumentCode(**{
#                     "classifier_id": model_id,
#                     "document_id": row["pk"],
#                     "code_id": row[handler.outcome_variable]
#                 })
#             )
#         ClassifierDocumentCode.objects.bulk_create(doc_codes)
#
#         print "Done processing chunk %i" % (int(i) + 1)
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