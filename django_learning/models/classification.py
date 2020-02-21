from __future__ import print_function
import copy, pandas, time

from django.db import models
from django.core.exceptions import ValidationError, NON_FIELD_ERRORS

from multiprocessing.pool import Pool

from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from django_commander.models import LoggedExtendedModel

from django_learning.utils.dataset_extractors import dataset_extractors
from django_learning.utils.decorators import require_model
from django_learning.utils.scoring import (
    apply_probability_threshold,
    get_probability_threshold_score_df,
)
from django_learning.models.learning import LearningModel, DocumentLearningModel

from pewtils import is_not_null, is_null, chunk_list
from django_pewtils import get_model


class ClassificationModel(LearningModel):

    probability_threshold = models.FloatField(default=None, null=True)

    # classifications = GenericRelation(Classification, related_query_name="classification_model")

    def _train_model(self, pipeline_steps, params, num_cores=1, **kwargs):

        results = super(ClassificationModel, self)._train_model(
            pipeline_steps, params, num_cores=num_cores, **kwargs
        )
        if self.probability_threshold:
            self.probability_threshold = None
            print("Removed previous probability threshold")
        self.save()

        return results

    def extract_dataset(self, refresh=False, **kwargs):

        super(ClassificationModel, self).extract_dataset(refresh=refresh, **kwargs)

        if self.parameters["model"].get("use_class_weights", False):

            # largest_code = self._get_largest_code()

            class_weights = {}
            # base_weight = df[df[self.dataset_extractor.outcome_column] == largest_code]['training_weight'].sum()
            # total_weight = df['training_weight'].sum()

            try:
                target_weight_per_class = self.dataset["training_weight"].sum() / float(
                    len(self.dataset[self.dataset_extractor.outcome_column].unique())
                )
            except KeyError:
                print("Huh...")
                import pdb

                pdb.set_trace()

            for c in self.dataset[self.dataset_extractor.outcome_column].unique():
                # class_weights[c] = float(df[df[self.outcome_column]==c]["training_weight"].sum()) / float(total_weight)
                # class_weights[c] = base_weight / float(df[df[self.dataset_extractor.outcome_column] == c]['training_weight'].sum())
                class_weights[c] = target_weight_per_class / float(
                    self.dataset[
                        self.dataset[self.dataset_extractor.outcome_column] == c
                    ]["training_weight"].sum()
                )

            # total_weight = sum(class_weights.values())
            # class_weights = {k: float(v) / float(total_weight) for k, v in class_weights.items()}
            print("Class weights: {}".format(class_weights))
            self.dataset["training_weight"] = self.dataset.apply(
                lambda x: x["training_weight"]
                * class_weights[x[self.dataset_extractor.outcome_column]],
                axis=1,
            )

        # if self.parameters["model"].get("use_class_weights", False):
        #     scale_pos_weight = df[df[self.dataset_extractor.outcome_column].astype(str) == str(largest_code)][
        #                            'training_weight'].sum() / \
        #                        df[df[self.dataset_extractor.outcome_column].astype(str) != str(largest_code)][
        #                            'training_weight'].sum()
        #     df.ix[df[self.dataset_extractor.outcome_column].astype(str) != str(
        #         largest_code), "training_weight"] *= scale_pos_weight

    @require_model
    def show_top_features(self, n=10):

        if hasattr(self.model, "best_estimator_"):
            model = self.model.best_estimator_
        else:
            model = self.model
        if hasattr(model, "named_steps"):
            steps = model.named_steps
        else:
            steps = model.steps

        feature_names = self.get_feature_names(model)
        class_labels = steps["model"].classes_

        top_features = {}
        if hasattr(steps["model"], "coef_"):
            try:
                coefs = list(steps["model"].coef_.todense().tolist())
            except AttributeError:
                coefs = list(steps["model"].coef_)
            if len(class_labels) == 2:
                top_features[0] = sorted(zip(coefs[0], feature_names))[:n]
                top_features[1] = sorted(zip(coefs[0], feature_names))[: -(n + 1) : -1]
            else:
                for i, class_label in enumerate(class_labels):
                    top_features[class_label] = sorted(zip(coefs[i], feature_names))[
                        -n:
                    ]
        elif hasattr(steps["model"], "feature_importances_"):
            top_features["n/a"] = sorted(
                zip(steps["model"].feature_importances_, feature_names)
            )[: -(n + 1) : -1]

        for class_label, top_n in top_features.items():
            print(class_label)
            for c, f in top_n:
                try:
                    print("\t%.4f\t\t%-15s" % (c, f))
                except:
                    print("Error: {}, {}".format(c, f))

    @require_model
    def describe_model(self):

        super(ClassificationModel, self).describe_model()
        self.show_top_features()

    def _get_positive_code(self):

        codes = self.dataset[self.dataset_extractor.outcome_column].unique()
        if len(codes) == 2:
            return [c for c in codes if str(c) != str(self._get_largest_code())][0]
        else:
            return None

    def print_test_prediction_report(self):

        results = self.get_test_prediction_results()

        report = classification_report(
            self.test_dataset[self.dataset_extractor.outcome_column],
            self.predict_dataset[self.dataset_extractor.outcome_column],
            sample_weight=self.test_dataset["sampling_weight"]
            if "sampling_weight" in self.test_dataset.columns
            and self._check_for_valid_weights(self.test_dataset["sampling_weight"])
            else None,
        )

        matrix = confusion_matrix(
            self.test_dataset[self.dataset_extractor.outcome_column],
            self.predict_dataset[self.dataset_extractor.outcome_column],
        )

        rows = []
        for row in report.split("\n"):
            row = row.strip().split()
            if len(row) == 7:
                row = row[2:]
            if len(row) == 5:
                rows.append(
                    {
                        "class": row[0],
                        "precision": row[1],
                        "recall": row[2],
                        "f1-score": row[3],
                        "support": row[4],
                    }
                )
        report = pandas.DataFrame(rows)

        print("Results: {}".format(results))
        print("Classification report: ")
        print(report)
        print("Confusion matrix: ")
        print(matrix)

    @require_model
    def get_test_prediction_results(self, refresh=False, only_load_existing=False):

        self.predict_dataset = None
        if is_not_null(self.test_dataset):
            self.predict_dataset = self.produce_prediction_dataset(
                self.test_dataset,
                cache_key="predict_test",
                refresh=refresh,
                only_load_existing=only_load_existing,
            )
            scores = self.compute_prediction_scores(
                self.test_dataset, predicted_df=self.predict_dataset
            )
            return scores

    @require_model
    def get_incorrect_predictions(
        self, correct_label_id=None, refresh=False, only_load_existing=False
    ):

        self.predict_dataset = None
        if is_not_null(self.dataset):
            self.predict_dataset = self.produce_prediction_dataset(
                self.dataset, refresh=refresh, only_load_existing=only_load_existing
            )
        merged = self.dataset.merge(
            self.predict_dataset[["document_id", "label_id"]],
            how="left",
            on="document_id",
            suffixes=("_test", "_predict"),
        )
        incorrect = merged[merged["label_id_test"] != merged["label_id_predict"]]
        if correct_label_id:
            incorrect = incorrect[incorrect["label_id_test"] == correct_label_id]

        return incorrect

    @require_model
    def get_cv_prediction_results(self, refresh=False, only_load_existing=False):

        print("Computing cross-fold predictions")
        _final_model = self.model
        if hasattr(self.model, "best_estimator_"):
            _final_model_best_estimator = self.model.best_estimator_
        else:
            _final_model_best_estimator = self.model

        dataset = copy.copy(self.train_dataset)

        all_fold_scores = []
        if refresh or not self.cv_folds:
            splitter = StratifiedKFold(
                self.parameters["model"].get("cv", 5), shuffle=True
            )
            self.cv_folds = [
                f
                for f in splitter.split(
                    dataset, dataset[self.dataset_extractor.outcome_column]
                )
            ]
            self.save()

        for i, folds in tqdm(enumerate(self.cv_folds), desc="Producing CV predictions"):
            fold_train_index, fold_test_index = folds
            # NOTE: KFold returns numerical index, so you need to remap it to the dataset index (which may not be numerical)
            fold_train_dataset = dataset.loc[
                pandas.Series(dataset.index).iloc[fold_train_index].values
            ]
            fold_test_dataset = dataset.loc[
                pandas.Series(dataset.index).iloc[fold_test_index].values
            ]

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
            fold_score_df = pandas.concat(
                [
                    all_fold_scores[0][["coder1", "coder2", "outcome_column"]],
                    fold_score_df.groupby(fold_score_df.index).mean(),
                ],
                axis=1,
            )
            return fold_score_df

    def find_probability_threshold(self, metric="precision_recall_min", save=False):

        """
        :param metric:
        :param save:
        :return:

        Iterates over thresholds and finds the one that maximizes the minimum of the specified metric
        between the test and CV fold prediction datasets
        """

        print("Scanning CV folds for optimal probability threshold")

        base_code = self._get_largest_code()
        if base_code:
            base_code = str(base_code)
        pos_code = self._get_positive_code()
        if pos_code:
            pos_code = str(pos_code)

        if save:
            self.probability_threshold = None
        if is_not_null(self.cv_folds):

            predict_dataset = self.produce_prediction_dataset(
                self.test_dataset,
                cache_key="predict_test",
                refresh=False,
                only_load_existing=True,
                ignore_probability_threshold=True,
            )
            if "probability" not in predict_dataset.columns:
                raise Exception("This model does not produce probabilities")
            test_threshold_scores = None
            if is_not_null(predict_dataset):
                test_threshold_scores = get_probability_threshold_score_df(
                    predict_dataset,
                    self.test_dataset,
                    outcome_column=self.dataset_extractor.outcome_column,
                    weight_column="sampling_weight"
                    if "sampling_weight" in predict_dataset.columns
                    and self._check_for_valid_weights(
                        predict_dataset["sampling_weight"]
                    )
                    else None,
                    base_code=base_code,
                    pos_code=pos_code,
                )

            dataset = copy.copy(self.train_dataset)

            all_fold_scores = []
            for i, folds in enumerate(self.cv_folds):
                fold_train_index, fold_test_index = folds
                # NOTE: KFold returns numerical index, so you need to remap it to the dataset index (which may not be numerical)
                fold_train_dataset = dataset.loc[
                    pandas.Series(dataset.index).iloc[fold_train_index].values
                ]  # self.dataset.ix[fold_train_index]
                fold_test_dataset = dataset.loc[
                    pandas.Series(dataset.index).iloc[fold_test_index].values
                ]  # self.dataset.ix[fold_test_index]

                fold_predict_dataset = self.produce_prediction_dataset(
                    fold_test_dataset,
                    cache_key="predict_fold_{}".format(i),
                    refresh=False,
                    only_load_existing=True,
                    ignore_probability_threshold=True,
                )
                # threshold = None
                if is_not_null(fold_predict_dataset):
                    fold_threshold_scores = get_probability_threshold_score_df(
                        fold_predict_dataset,
                        fold_test_dataset,
                        outcome_column=self.dataset_extractor.outcome_column,
                        weight_column="sampling_weight"
                        if "sampling_weight" in fold_predict_dataset.columns
                        and self._check_for_valid_weights(
                            fold_predict_dataset["sampling_weight"]
                        )
                        else None,
                        base_code=base_code,
                        pos_code=pos_code,
                    )
                    if is_not_null(test_threshold_scores):
                        fold_threshold_scores[metric] = [
                            min(list(x))
                            for x in zip(
                                test_threshold_scores[metric],
                                fold_threshold_scores[metric],
                            )
                        ]
                    all_fold_scores.append(fold_threshold_scores)

                #     # if is_not_null(test_threshold_scores):
                #     #     fold_threshold_scores[metric] = [min(list(x)) for x in zip(test_threshold_scores[metric], fold_threshold_scores[metric])]
                #     threshold = get_probability_threshold_from_score_df(
                #         fold_threshold_scores,
                #         metric=metric
                #     )
                #     fold_predict_dataset = apply_probability_threshold(
                #         fold_predict_dataset,
                #         threshold,
                #         outcome_column=self.dataset_extractor.outcome_column,
                #         base_code=base_code,
                #         pos_code=pos_code
                #     )
                #
                # if is_not_null(fold_predict_dataset):
                #     fold_scores = self.compute_prediction_scores(fold_test_dataset, predicted_df=fold_predict_dataset)
                #     fold_scores['probability_threshold'] = threshold
                # else:
                #     fold_scores = None
                # all_fold_scores.append(fold_scores)

            if any([is_null(f) for f in all_fold_scores]):
                print(
                    "You don't have CV predictions saved in the cache; please run 'get_cv_prediction_results' first"
                )
                if save:
                    self.set_probability_threshold(None)
                return None
            else:
                fold_score_df = pandas.concat(all_fold_scores).fillna(0.0)
                threshold = (
                    fold_score_df.groupby(["threshold", "outcome_column"])
                    .mean()[metric]
                    .sort_values(ascending=False)
                    .index[0][0]
                )
                if save:
                    self.set_probability_threshold(threshold)
                return threshold

            # if any([is_null(f) for f in all_fold_scores]):
            #     print "You don't have CV predictions saved in the cache; please run 'get_cv_prediction_results' first"
            #     return None
            # else:
            #     fold_score_df = pandas.concat(all_fold_scores)
            #     fold_score_df = pandas.concat([
            #         all_fold_scores[0][["coder1", "coder2", "outcome_column"]],
            #         fold_score_df.groupby(fold_score_df.index).mean()
            #     ], axis=1)
            #     threshold = fold_score_df['probability_threshold'].mean()
            #     if save:
            #         self.set_probability_threshold(threshold)
            #     return threshold
        else:
            if save:
                self.set_probability_threshold(None)
            return None

    def set_probability_threshold(self, threshold):

        self.probability_threshold = threshold
        self.save()

    def apply_model(
        self,
        data,
        keep_cols=None,
        clear_temp_cache=True,
        disable_probability_threshold_warning=False,
    ):

        results = super(ClassificationModel, self).apply_model(
            data, keep_cols=keep_cols, clear_temp_cache=clear_temp_cache
        )
        if self.probability_threshold and not disable_probability_threshold_warning:
            print(
                "Warning: because 'apply_model' is used by model prediction dataset extractors, which cache their results, "
            )
            print(
                "probability thresholds are not applied, for the sake of efficiency.  If you wish to apply the threshold, "
            )
            print(
                "use 'produce_prediction_dataset' and pass 'ignore_probability_threshold=False' or manually pass the results "
            )
            print("to 'django_learning.utils.scoring.apply_probabily_threshold'")
        return results

    @require_model
    def produce_prediction_dataset(
        self,
        df_to_predict,
        cache_key=None,
        refresh=False,
        only_load_existing=False,
        ignore_probability_threshold=False,
    ):

        base_code = self._get_largest_code()
        if base_code:
            base_code = str(base_code)
        pos_code = self._get_positive_code()
        if pos_code:
            pos_code = str(pos_code)

        predicted_df = super(ClassificationModel, self).produce_prediction_dataset(
            df_to_predict,
            cache_key=cache_key,
            refresh=refresh,
            only_load_existing=only_load_existing,
            disable_probability_threshold_warning=(not ignore_probability_threshold),
        )
        if is_not_null(predicted_df):
            if not ignore_probability_threshold:
                if not self.probability_threshold:
                    print("No probability threshold is currently set, skipping")
                else:
                    predicted_df = apply_probability_threshold(
                        predicted_df,
                        self.probability_threshold,
                        outcome_column=self.dataset_extractor.outcome_column,
                        base_code=base_code,
                        pos_code=pos_code,
                    )
            elif self.probability_threshold:
                print(
                    "Probability threshold exists ({}) but you've said to ignore it".format(
                        self.probability_threshold
                    )
                )
        return predicted_df


class DocumentClassificationModel(ClassificationModel, DocumentLearningModel):
    def extract_dataset(self, refresh=False, **kwargs):

        super(DocumentClassificationModel, self).extract_dataset(
            refresh=refresh, **kwargs
        )
        for additional_weight in ["balancing_weight"]:
            if additional_weight in self.dataset.columns:
                print("Mixing {} into the training weights".format(additional_weight))
                self.dataset["training_weight"] = (
                    self.dataset["training_weight"] * self.dataset[additional_weight]
                )
        self.document_types = self.dataset["document_type"].unique()

    def _get_all_document_types(self):

        return [
            f.name
            for f in get_model(
                "Document", app_name="django_learning"
            ).get_parent_relations()
        ]

    @require_model
    def apply_model_to_documents(
        self, documents, save=True, document_filters=None, refresh=False
    ):

        extractor = dataset_extractors["raw_document_dataset"](
            document_ids=list(documents.values_list("pk", flat=True)),
            document_filters=document_filters,
        )
        dataset = extractor.extract(refresh=refresh)

        predictions = pandas.DataFrame()
        if len(dataset) > 0:

            predictions = self.produce_prediction_dataset(
                dataset,
                cache_key=extractor.get_hash(),
                refresh=refresh,
                only_load_existing=False,
            )

            if save:

                print("{} predicted documents".format(len(predictions)))
                Classification.objects.filter(
                    document_id__in=predictions["document_id"],
                    classification_model=self,
                ).delete()
                classifications = []
                for index, row in predictions.iterrows():
                    classifications.append(
                        Classification(
                            **{
                                "document_id": row["document_id"],
                                "label_id": row[self.dataset_extractor.outcome_column],
                                "classification_model": self,
                                "probability": row.get("probability", None),
                            }
                        )
                    )
                print("{} classifications to create".format(len(classifications)))
                Classification.objects.bulk_create(classifications)

                return None

        return predictions

    @require_model
    def apply_model_to_documents_multiprocessed(
        self,
        document_ids,
        save=True,
        document_filters=None,
        refresh=False,
        num_cores=2,
        chunk_size=1000,
    ):

        print("Processing {} documents".format(len(document_ids)))

        pool = Pool(processes=num_cores)
        results = []
        for i, chunk in enumerate(chunk_list(document_ids, chunk_size)):
            print("Creating chunk {}".format(i))
            if num_cores == 1:
                func = pool.apply
            else:
                func = pool.apply_async
            result = func(
                _process_document_chunk,
                args=(self.pk, chunk, i, save, document_filters, refresh),
            )
            results.append(result)
        pool.close()
        pool.join()

        try:
            results = [r.get() for r in results]
            results = [r for r in results if is_not_null(r)]
            return pandas.concat(results)
        except (ValueError, AttributeError):
            return None

    @require_model
    def apply_model_to_frame(
        self,
        save=True,
        document_filters=None,
        refresh=False,
        num_cores=2,
        chunk_size=1000,
    ):

        doc_ids = set(list(self.sampling_frame.documents.values_list("pk", flat=True)))
        if not refresh:
            existing_doc_ids = set(
                list(self.classifications.values_list("document_id", flat=True))
            )
            doc_ids = doc_ids.difference(existing_doc_ids)
        self.apply_model_to_documents_multiprocessed(
            list(doc_ids),
            save=save,
            document_filters=document_filters,
            refresh=True,
            num_cores=num_cores,
            chunk_size=chunk_size,
        )

    def update_classifications_with_probability_threshold(self):

        base_code = self._get_largest_code()
        pos_code = self._get_positive_code()
        switch_to_pos = self.classifications.filter(label_id=base_code).filter(
            probability__gte=self.probability_threshold
        )
        switch_to_neg = self.classifications.filter(label_id=pos_code).filter(
            probability__lt=self.probability_threshold
        )
        switch_to_pos.update(label_id=pos_code)
        switch_to_neg.update(label_id=base_code)


def _process_document_chunk(model_id, chunk, i, save, document_filters, refresh):

    import sys, traceback
    from django_learning.models import DocumentClassificationModel
    from django_pewtils import get_model, reset_django_connection
    from pewtils import is_not_null
    from django.conf import settings

    reset_django_connection(settings.SITE_NAME)

    try:

        documents = get_model("Document", app_name="django_learning").objects.filter(
            pk__in=chunk
        )
        model = DocumentClassificationModel.objects.get(pk=model_id)
        model.load_model(only_load_existing=True)
        if is_null(model.model):
            time.sleep(5.0)
            model.load_model(only_load_existing=True)
        if is_not_null(model.model):

            predictions = model.apply_model_to_documents(
                documents, save=save, document_filters=document_filters, refresh=refresh
            )

            print("Done processing chunk %i" % (int(i) + 1))

        else:
            raise Exception("Couldn't load the model, aborting this chunk")

        return predictions

    except Exception as e:

        print(e)
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print(exc_type)
        print(exc_value)
        print(exc_traceback)
        traceback.print_exc(exc_traceback)
        raise


class Classification(LoggedExtendedModel):

    document = models.ForeignKey(
        "django_learning.Document",
        on_delete=models.CASCADE,
        related_name="classifications",
    )
    label = models.ForeignKey(
        "django_learning.Label",
        on_delete=models.CASCADE,
        related_name="classifications",
    )

    classification_model = models.ForeignKey(
        "django_learning.DocumentClassificationModel",
        related_name="classifications",
        on_delete=models.CASCADE,
    )

    probability = models.FloatField(
        null=True, help_text="The probability of the assigned label, if applicable"
    )

    def validate_unique(self, *args, **kwargs):
        super(Classification, self).validate_unique(*args, **kwargs)
        if (
            not self.pk
            and not self.label.question.multiple
            and self.model.objects.filter(label__question=self.label.question)
            .filter(document=self.document)
            .exists()
        ):
            raise ValidationError(
                {
                    NON_FIELD_ERRORS: [
                        "Classification with the same variable already exists"
                    ]
                }
            )

    def __repr__(self):
        return "<Classification label={0}, document={1}>".format(
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
