# from logos.learning.utils.decorators import require_model
# from .basic_handler import SupervisedLearningHandler
#
#
# class RegressionHandler(SupervisedLearningHandler):
#
#
#     pipeline_folders = ['regression']
#
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
#             ))[:-(n+1):-1]
#         elif hasattr(self.model.best_estimator_.named_steps['model'], "feature_importances_"):
#             top_features = sorted(zip(
#                 self.model.best_estimator_.named_steps['model'].feature_importances_,
#                 feature_names
#             ))[:-(n+1):-1]
#         for c, f in top_features:
#             print "\t%.4f\t\t%-15s" % (c, f)
#
