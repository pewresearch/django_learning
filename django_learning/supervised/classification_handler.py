import pandas
from sklearn.metrics import classification_report, confusion_matrix

from django_learning.utils.decorators import require_model
from django_learning.supervised.basic_handler import SupervisedLearningHandler


class ClassificationHandler(SupervisedLearningHandler):


    pipeline_folders = ['classification']


    @require_model
    def show_top_features(self, n=10):

        if hasattr(self.model.best_estimator_, "named_steps"): steps = self.model.best_estimator_.named_steps
        else: steps = self.model.best_estimator_.steps

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
                ))[:-(n+1):-1]
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
            ))[:-(n+1):-1]

        for class_label, top_n in top_features.iteritems():
            print class_label
            for c, f in top_n:
                try:
                    print "\t%.4f\t\t%-15s" % (c, f)
                except:
                    print "Error: {}, {}".format(c, f)


    @require_model
    def print_report(self):

        super(ClassificationHandler, self).print_report()

        print "Detailed classification report:"
        print classification_report(self.test_y, self.predict_y, sample_weight=self.test_x['sampling_weight'] if self.parameters["model"].get("use_sample_weights", False) else None)

        print "Confusion matrix:"
        print confusion_matrix(self.test_y, self.predict_y)


    @require_model
    def get_incorrect_predictions(self, actual_code=None):

        df = pandas.concat([self.test_y, self.test_x], axis=1)
        df['prediction'] = self.predict_y
        df = df[df['code_id']!=df['prediction']]
        if actual_code:
            df = df[df['code_id']==actual_code]
        return df


    @require_model
    def get_report_results(self):

        rows = []
        report = classification_report(self.test_y, self.predict_y, sample_weight=self.test_x['sampling_weight'] if self.parameters["model"].get("use_sample_weights", False) else None)
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