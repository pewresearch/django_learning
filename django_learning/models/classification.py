import importlib, copy, pandas, numpy

from django.db import models
from django.db.models import Q
from django.contrib.postgres.fields import ArrayField
from django.conf import settings
from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
from django.contrib.contenttypes.models import ContentType
from django.core.exceptions import ValidationError, NON_FIELD_ERRORS

from picklefield.fields import PickledObjectField
from langdetect import detect
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from statsmodels.stats.inter_rater import cohens_kappa
from multiprocessing.pool import Pool

from sklearn.cross_validation import train_test_split, KFold
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss, make_scorer, mean_squared_error, r2_score, matthews_corrcoef, accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import ttest_ind
from tqdm import tqdm

from django_commander.models import LoggedExtendedModel

from django_learning.utils import get_document_types, get_pipeline_repr, get_param_repr
from django_learning.utils.pipelines import pipelines
from django_learning.utils.dataset_extractors import dataset_extractors
from django_learning.utils.decorators import require_training_data, require_model, temp_cache_wrapper
from django_learning.utils.feature_extractors import BasicExtractor
from django_learning.utils.scoring import find_probability_threshold, apply_probability_threshold, get_probability_threshold_score_df, get_probability_threshold_from_score_df
from django_learning.utils.models import models as learning_models
from django_learning.models.learning import LearningModel, DocumentLearningModel

from pewtils import is_not_null, is_null, decode_text, recursive_update, chunker
from pewtils.django import get_model, CacheHandler, reset_django_connection_wrapper, django_multiprocessor
from pewtils.sampling import compute_sample_weights_from_frame, compute_balanced_sample_weights
from pewtils.stats import wmom


class ClassificationModel(LearningModel):

    probability_threshold = models.FloatField(default=None, null=True)

    # classifications = GenericRelation(Classification, related_query_name="classification_model")

    def _train_model(self, pipeline_steps, params, num_cores=1, **kwargs):

        results = super(ClassificationModel, self)._train_model(pipeline_steps, params, num_cores=num_cores, **kwargs)
        if self.probability_threshold:
            self.probability_threshold = None
            print "Removed previous probability threshold"
        self.save()

        return results

    def _get_training_dataset(self):

        df = super(ClassificationModel, self)._get_training_dataset()

        if self.parameters["model"].get("use_class_weights", False):

            largest_code = self._get_largest_code()

            class_weights = {}
            # base_weight = df[df[self.dataset_extractor.outcome_column] == largest_code]['training_weight'].sum()
            # total_weight = df['training_weight'].sum()

            target_weight_per_class = df['training_weight'].sum() / float(len(df[self.dataset_extractor.outcome_column].unique()))

            for c in df[self.dataset_extractor.outcome_column].unique():
                # class_weights[c] = float(df[df[self.outcome_column]==c]["training_weight"].sum()) / float(total_weight)
                # class_weights[c] = base_weight / float(df[df[self.dataset_extractor.outcome_column] == c]['training_weight'].sum())
                class_weights[c] = target_weight_per_class / float(df[df[self.dataset_extractor.outcome_column] == c]['training_weight'].sum())

            # total_weight = sum(class_weights.values())
            # class_weights = {k: float(v) / float(total_weight) for k, v in class_weights.items()}
            print "Class weights: {}".format(class_weights)
            df["training_weight"] = df.apply(lambda x: x['training_weight']*class_weights[x[self.dataset_extractor.outcome_column]], axis=1)

        # if self.parameters["model"].get("use_class_weights", False):
        #     scale_pos_weight = df[df[self.dataset_extractor.outcome_column].astype(str) == str(largest_code)][
        #                            'training_weight'].sum() / \
        #                        df[df[self.dataset_extractor.outcome_column].astype(str) != str(largest_code)][
        #                            'training_weight'].sum()
        #     df.ix[df[self.dataset_extractor.outcome_column].astype(str) != str(
        #         largest_code), "training_weight"] *= scale_pos_weight

        return df

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
            try: coefs = list(steps['model'].coef_.todense().tolist())
            except AttributeError: coefs = list(steps['model'].coef_)
            if len(class_labels) == 2:
                top_features[0] = sorted(zip(
                    coefs[0],
                    feature_names
                ))[:n]
                top_features[1] = sorted(zip(
                    coefs[0],
                    feature_names
                ))[:-(n + 1):-1]
            else:
                for i, class_label in enumerate(class_labels):
                    top_features[class_label] = sorted(zip(
                        coefs[i],
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
    def describe_model(self):

        super(ClassificationModel, self).describe_model()
        self.show_top_features()

    def print_test_prediction_report(self):

        results = self.get_test_prediction_results()

        report = classification_report(
            self.test_dataset[self.dataset_extractor.outcome_column],
            self.predict_dataset[self.dataset_extractor.outcome_column],
            sample_weight=self.test_dataset["sampling_weight"] if "sampling_weight" in self.test_dataset.columns and
                                                                  self.parameters["model"].get("use_sample_weights",
                                                                                               False) else None
        )

        matrix = confusion_matrix(
            self.test_dataset[self.dataset_extractor.outcome_column],
            self.predict_dataset[self.dataset_extractor.outcome_column]
        )

        rows = []
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
        report = pandas.DataFrame(rows)

        print "Results: {}".format(results)
        print "Classification report: "
        print report
        print "Confusion matrix: "
        print matrix

    @require_model
    def get_test_prediction_results(self, refresh=False, only_get_existing=False):

        self.predict_dataset = None
        if is_not_null(self.test_dataset):
            self.predict_dataset = self.produce_prediction_dataset(self.test_dataset, cache_key="predict_main",
                                                                   refresh=refresh,
                                                                   only_get_existing=only_get_existing)
            scores = self.compute_prediction_scores(self.test_dataset, predicted_df=self.predict_dataset)
            return scores

    @require_model
    def get_cv_prediction_results(self, refresh=False, only_get_existing=False):

        print "Computing cross-fold predictions"
        _final_model = self.model
        _final_model_best_estimator = self.model.best_estimator_
        dataset = self._get_training_dataset()

        all_fold_scores = []
        if refresh or not self.cv_folds:
            self.cv_folds = [f for f in KFold(len(dataset.index), n_folds=self.parameters["model"].get("cv", 5), shuffle=True)]
            self.save()

        for i, folds in tqdm(enumerate(self.cv_folds), desc="Producing CV predictions"):
            fold_train_index, fold_test_index = folds
            # NOTE: KFold returns numerical index, so you need to remap it to the dataset index (which may not be numerical)
            fold_train_dataset = dataset.ix[pandas.Series(dataset.index).iloc[fold_train_index].values]  # self.dataset.ix[fold_train_index]
            fold_test_dataset = dataset.ix[pandas.Series(dataset.index).iloc[fold_test_index].values]  # self.dataset.ix[fold_test_index]

            fold_predict_dataset = None
            if not refresh:
                fold_predict_dataset = self.produce_prediction_dataset(fold_test_dataset,
                                                                       cache_key="predict_fold_{}".format(i),
                                                                       refresh=False, only_get_existing=True)
            if is_null(fold_predict_dataset) and not only_get_existing:

                fit_params = self._get_fit_params(fold_train_dataset)
                self.model = _final_model_best_estimator.fit(
                    fold_train_dataset,
                    fold_train_dataset[self.dataset_extractor.outcome_column],
                    **fit_params
                )
                fold_predict_dataset = self.produce_prediction_dataset(fold_test_dataset,
                                                                       cache_key="predict_fold_{}".format(i),
                                                                       refresh=refresh)

            if is_not_null(fold_predict_dataset):
                fold_scores = self.compute_prediction_scores(fold_test_dataset, predicted_df=fold_predict_dataset)
            else:
                fold_scores = None
            all_fold_scores.append(fold_scores)

        self.model = _final_model
        if any([is_null(f) for f in all_fold_scores]):
            return None
        else:
            fold_score_df = pandas.concat(all_fold_scores)
            fold_score_df = pandas.concat([
                all_fold_scores[0][["coder1", "coder2", "outcome_column"]],
                fold_score_df.groupby(fold_score_df.index).mean()
            ], axis=1)
            return fold_score_df

    def find_probability_threshold(self, metric="precision_recall_min", save=False):

        """
        :param metric:
        :param save:
        :return:

        Iterates over thresholds and finds the one that maximizes the minimum of the specified metric
        between the test and CV fold prediction datasets
        """

        print "Scanning CV folds for optimal probability threshold"

        self.probability_threshold = None
        if is_not_null(self.cv_folds):

            predict_dataset = self.produce_prediction_dataset(
                self.test_dataset,
                cache_key="predict_main",
                refresh=False,
                only_get_existing=True,
                ignore_probability_threshold=True
            )
            test_threshold_scores = None
            if is_not_null(predict_dataset):
                test_threshold_scores = get_probability_threshold_score_df(
                    predict_dataset,
                    self.test_dataset,
                    outcome_column=self.dataset_extractor.outcome_column,
                    metric=metric,
                    weight_column="sampling_weight" if "sampling_weight" in predict_dataset.columns else None
                )

            dataset = self._get_training_dataset()

            all_fold_scores = []
            for i, folds in enumerate(self.cv_folds):
                fold_train_index, fold_test_index = folds
                # NOTE: KFold returns numerical index, so you need to remap it to the dataset index (which may not be numerical)
                fold_train_dataset = dataset.ix[pandas.Series(dataset.index).iloc[fold_train_index].values]  # self.dataset.ix[fold_train_index]
                fold_test_dataset = dataset.ix[pandas.Series(dataset.index).iloc[fold_test_index].values]  # self.dataset.ix[fold_test_index]

                fold_predict_dataset = self.produce_prediction_dataset(
                    fold_test_dataset,
                    cache_key="predict_fold_{}".format(i),
                    refresh=False,
                    only_get_existing=True,
                    ignore_probability_threshold=True
                )
                threshold = None
                if is_not_null(fold_predict_dataset):
                    fold_threshold_scores = get_probability_threshold_score_df(
                        fold_predict_dataset,
                        fold_test_dataset,
                        outcome_column=self.dataset_extractor.outcome_column,
                        metric=metric,
                        weight_column="sampling_weight" if "sampling_weight" in fold_predict_dataset.columns else None
                    )
                    if is_not_null(test_threshold_scores):
                        fold_threshold_scores[metric] = [min(list(x)) for x in zip(test_threshold_scores[metric], fold_threshold_scores[metric])]
                    threshold = get_probability_threshold_from_score_df(
                        fold_threshold_scores,
                        metric=metric,
                        base_id=str(self._get_largest_code())
                    )
                    fold_predict_dataset = apply_probability_threshold(
                        fold_predict_dataset,
                        threshold,
                        outcome_column=self.dataset_extractor.outcome_column,
                        base_id=str(self._get_largest_code())
                    )

                if is_not_null(fold_predict_dataset):
                    fold_scores = self.compute_prediction_scores(fold_test_dataset, predicted_df=fold_predict_dataset)
                    fold_scores['probability_threshold'] = threshold
                else:
                    fold_scores = None
                all_fold_scores.append(fold_scores)

        if any([is_null(f) for f in all_fold_scores]):
            print "You don't have CV predictions saved in the cache; please run 'get_cv_prediction_results' first"
            return None
        else:
            fold_score_df = pandas.concat(all_fold_scores)
            fold_score_df = pandas.concat([
                all_fold_scores[0][["coder1", "coder2", "outcome_column"]],
                fold_score_df.groupby(fold_score_df.index).mean()
            ], axis=1)
            threshold = fold_score_df['probability_threshold'].mean()
            if save:
                self.set_probability_threshold(threshold)
            return threshold

    def set_probability_threshold(self, threshold):

        self.probability_threshold = threshold
        self.save()

    def apply_model(self, data, keep_cols=None, clear_temp_cache=True):

        results = super(ClassificationModel, self).apply_model(data, keep_cols=keep_cols, clear_temp_cache=clear_temp_cache)
        if self.probability_threshold:
            print "Warning: because 'apply_model' is used by model prediction dataset extractors, which cache their results, "
            print "probability thresholds are not applied, for the sake of efficiency.  If you wish to apply the threshold, "
            print "use 'produce_prediction_dataset' and pass 'ignore_probability_threshold=False' or manually pass the results "
            print "to 'django_learning.utils.scoring.apply_probabily_threshold'"
        return results

    @require_model
    def produce_prediction_dataset(self, df_to_predict, cache_key=None, refresh=False, only_get_existing=False, ignore_probability_threshold=False):

        predicted_df = super(ClassificationModel, self).produce_prediction_dataset(df_to_predict, cache_key=cache_key, refresh=refresh, only_get_existing=only_get_existing)
        if is_not_null(predicted_df):
            if not ignore_probability_threshold:
                if not self.probability_threshold:
                    print "No probability threshold is currently set, skipping"
                else:
                    predicted_df = apply_probability_threshold(
                        predicted_df,
                        self.probability_threshold,
                        outcome_column=self.dataset_extractor.outcome_column,
                        base_id=str(self._get_largest_code())
                    )
            elif self.probability_threshold:
                print "Probability threshold exists ({}) but you've said to ignore it".format(self.probability_threshold)
        return predicted_df

    # @require_model
    # def get_incorrect_predictions(self, actual_code=None):
    #
    #     df = pandas.concat([self.test_y, self.test_x], axis=1)
    #     df['prediction'] = self.predict_y
    #     df = df[df[self.outcome_column] != df['prediction']]
    #     if actual_code:
    #         df = df[df[self.outcome_column] == actual_code]
    #     return df


class DocumentClassificationModel(ClassificationModel, DocumentLearningModel):

    def extract_dataset(self, refresh=False, **kwargs):

        super(DocumentClassificationModel, self).extract_dataset(refresh=refresh, **kwargs)
        for additional_weight in ["balancing_weight"]:
            if additional_weight in self.dataset.columns:
                print "Mixing {} into the training weights".format(additional_weight)
                self.dataset["training_weight"] = self.dataset["training_weight"] * self.dataset[additional_weight]
        self.document_types = self.dataset["document_type"].unique()

    def _get_all_document_types(self):

        return [f.name for f in get_model("Document", app_name="django_learning").get_parent_relations()]

    @require_model
    def apply_model_to_documents(
        self,
        documents,
        save=True,
        document_filters=None,
        refresh_document_dataset=False,
        refresh_predictions=False
    ):

        extractor = dataset_extractors["raw_document_dataset"](
            document_ids=list(documents.values_list("pk", flat=True)),
            document_filters=document_filters
        )
        dataset = extractor.extract(refresh=refresh_document_dataset)

        predictions = pandas.DataFrame()
        if len(dataset) > 0:

            predictions = self.produce_prediction_dataset(dataset, cache_key=extractor.get_hash(), refresh=refresh_predictions, only_get_existing=False)

            if save:

                print "{} predicted documents".format(len(predictions))
                Classification.objects.filter(
                    document_id__in=predictions["document_id"],
                    classification_model=self
                ).delete()
                classifications = []
                for index, row in predictions.iterrows():
                    classifications.append(
                        Classification(**{
                            "document_id": row["document_id"],
                            "label_id": row[self.dataset_extractor.outcome_column],
                            "classification_model": self,
                            "probability": row.get("probability", None)
                        })
                    )
                print "{} classifications to create".format(len(classifications))
                Classification.objects.bulk_create(classifications)

                return None

        return predictions

    @require_model
    def apply_model_to_documents_multiprocessed(
        self,
        documents,
        save=True,
        document_filters=None,
        refresh_document_dataset=False,
        refresh_predictions=False,
        num_cores=2,
        chunk_size=1000
    ):

        try:
            document_ids = list(documents.values_list("pk", flat=True))
        except:
            document_ids = [getattr(d, "pk") for d in documents]

        print "Processing {} documents".format(len(document_ids))

        pool = Pool(processes=num_cores)
        results = []
        for i, chunk in enumerate(chunker(document_ids, chunk_size)):
            print "Creating chunk {}".format(i)
            if num_cores == 1: func = pool.apply
            else: func = pool.apply_async
            result = func(_process_document_chunk, args=(
                self.pk,
                chunk,
                i,
                save,
                document_filters,
                refresh_document_dataset,
                refresh_predictions
            ))
            results.append(result)
        pool.close()
        pool.join()

        try:
            results = [r.get() for r in results]
            results = [r for r in results if is_not_null(r)]
            return pandas.concat(results)
        except AttributeError:
            return None

    @require_model
    def apply_model_to_frame(
        self,
        save=True,
        document_filters=None,
        refresh_document_dataset=False,
        refresh_predictions=False,
        num_cores=2,
        chunk_size=1000
    ):

        docs = self.sampling_frame.documents.all()
        self.apply_model_to_documents_multiprocessed(
            docs,
            save=save,
            document_filters=document_filters,
            refresh_document_dataset=refresh_document_dataset,
            refresh_predictions=refresh_predictions,
            num_cores=num_cores,
            chunk_size=chunk_size
        )


def _process_document_chunk(
    model_id,
    chunk,
    i,
    save,
    document_filters,
    refresh_document_dataset,
    refresh_predictions
):

    import sys, traceback
    from django_learning.models import DocumentClassificationModel
    from pewtils.django import get_model, reset_django_connection
    from django.conf import settings
    reset_django_connection(settings.SITE_NAME)

    try:

        documents = get_model("Document", app_name="django_learning").objects.filter(pk__in=chunk)
        model = DocumentClassificationModel.objects.get(pk=model_id)
        model.load_model(only_load_existing=True)
        predictions = model.apply_model_to_documents(
            documents,
            save=save,
            document_filters=document_filters,
            refresh_document_dataset=refresh_document_dataset,
            refresh_predictions=refresh_predictions
        )

        print "Done processing chunk %i" % (int(i) + 1)

        return predictions

    except Exception as e:

        print e
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print exc_type
        print exc_value
        print exc_traceback
        traceback.print_exc(exc_traceback)
        raise


class Classification(LoggedExtendedModel):

    document = models.ForeignKey("django_learning.Document", related_name="classifications")
    label = models.ForeignKey("django_learning.Label", related_name="classifications")

    classification_model = models.ForeignKey("django_learning.DocumentClassificationModel", related_name="classifications")

    probability = models.FloatField(null=True, help_text="The probability of the assigned label, if applicable")

    def validate_unique(self, *args, **kwargs):
       super(Classification, self).validate_unique(*args, **kwargs)
       if not self.pk:
           if not self.label.question.multiple:
               if self.model.objects\
                       .filter(label__question=self.label.question)\
                       .filter(document=self.document)\
                       .exists():
                   raise ValidationError(
                       {
                           NON_FIELD_ERRORS: [
                               'Classification with the same variable already exists'
                           ],
                       }
                   )

    def __repr__(self):
        return "<Classification label={0}, document={2}>".format(
            self.label, self.document
        )


# ah, okay, so originally you had spec-ed out the model below
# but... DocumentClassificationModels are the only thing that's every going to make
# Classifications... on Documents... so... no need for a generic foreign key, right?
# class Classification(LoggedExtendedModel):
#
#     document = models.ForeignKey("django_learning.Document", related_name="classifications")
#     label = models.ForeignKey("django_learning.Label", related_name="classifications")
#
#     # classifier = models.ForeignKey("django_learning.Classifier", related_name="classifications")
#     content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
#     object_id = models.PositiveIntegerField()
#     model = GenericForeignKey('content_type', 'object_id')
#
#     probability = models.FloatField(null=True, help_text="The probability of the assigned label, if applicable")
#
#     # def validate_unique(self, *args, **kwargs):
#     #        super(Classification, self).validate_unique(*args, **kwargs)
#     #        if not self.id:
#     #            if not self.label.question.multiple:
#     #                if self.model.objects\
#     #                        .filter(label__question=self.label.question)\
#     #                        .filter(document=self.document)\
#     #                        .exists():
#     #                    raise ValidationError(
#     #                        {
#     #                            NON_FIELD_ERRORS: [
#     #                                'Classification with the same variable already exists'
#     #                            ],
#     #                        }
#     #                    )
#
#     def __repr__(self):
#         return "<Classification label={0}, document={2}>".format(
#             self.label, self.document
#         )