# import importlib, copy, pandas, numpy
#
# from django.db import models
# from django.db.models import Q
# from django.contrib.postgres.fields import ArrayField
# from django.conf import settings
# from django.contrib.contenttypes.fields import GenericForeignKey, GenericRelation
# from django.contrib.contenttypes.models import ContentType
#
# from picklefield.fields import PickledObjectField
# from langdetect import detect
# from abc import abstractmethod
# from collections import OrderedDict, defaultdict
# from statsmodels.stats.inter_rater import cohens_kappa
#
# from sklearn.cross_validation import train_test_split, KFold
# from sklearn.grid_search import GridSearchCV
# from sklearn.metrics import f1_score, precision_score, recall_score, brier_score_loss, make_scorer, mean_squared_error, r2_score, matthews_corrcoef, accuracy_score, f1_score, roc_auc_score
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn.metrics import classification_report, confusion_matrix
# from scipy.stats import ttest_ind
#
# from django_commander.models import LoggedExtendedModel
#
# from django_learning.utils import get_document_types, get_pipeline_repr, get_param_repr
# from django_learning.utils.pipelines import pipelines
# from django_learning.utils.training_data_extractors import training_data_extractors
# from django_learning.utils.decorators import require_training_data, require_model, temp_cache_wrapper
# from django_learning.utils.feature_extractors import BasicExtractor
# from django_learning.utils.models import models as learning_models
# from django_learning.models.learning import LearningModel, DocumentLearningModel
#
# from pewtils import is_not_null, is_null, decode_text, recursive_update
# from django_pewtils import get_model, CacheHandler
# from pewtils.sampling import compute_sample_weights_from_frame, compute_balanced_sample_weights
# from pewtils.stats import wmom
#
#
# class Regression(LoggedExtendedModel):
#
#     document = models.ForeignKey("django_learning.Document", related_name="classifications")
#     value = models.FloatField()
#
#     # classifier = models.ForeignKey("django_learning.Classifier", related_name="classifications")
#     content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
#     object_id = models.PositiveIntegerField()
#     model = GenericForeignKey('content_type', 'object_id')
#
#     probability = models.FloatField(null=True)
#
#     def __repr__(self):
#         return "<Regression value={0}, document={2}>".format(
#             self.value, self.document
#         )
#
#
# class RegressionModel(LearningModel):
#
#     DATASET_MODELS = models.Q(app_label="django_learning", model="extractor_dataset") | \
#                      models.Q(app_label="django_learning", model="code_dataset")
#     content_type = models.ForeignKey(ContentType, limit_choices_to=DATASET_MODELS)
#     object_id = models.PositiveIntegerField()
#     training_dataset = generic.GenericForeignKey('content_type', 'object_id')
#
#     regressions = GenericRelation(Regression)
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
#         super(RegressionModel, self).print_report()
#
#         self.show_top_features()
#
#
# class CodeRegressionModel(RegressionModel, CodeLearningModel):
#
#     pass
