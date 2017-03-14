Scoring functions
==================

Scoring functions are used to calculate a model's performance. They are specified in config files that should be placed
in one of your ``settings.DJANGO_LEARNING_SCORING_FUNCTIONS`` folders, and the file should contain a ``scorer``
function that accepts positional arguments for an array of true values and predicted values, and a keyword
argument for sampling weights.

.. code:: python

    def scorer(y_true, y_pred, sample_weight=None):
        # Do something here
        score = 1.0
        return score

Scoring functions are primarily used in Django Learning pipelines, specified in the "model" section of the pipeline
under the parameter "scoring_function". (See Pipelines for more.)

Higher values should indicate better model performance.

Built-in scoring functions
---------------------------

cohens_kappa
*************
Returns Cohen's Kappa based on ``sklearn.metrics.cohen_kappa_score``

matthews_corrcoef
**************************

Returns Matthew's correlation coefficient based on ``sklearn.metrics.matthews_corrcoef``, using the least-frequent
value in ``y_true`` as the positive class, and all other values as negative.

maxmin
***********

Returns the minimum of [precision, recall]. Useful for trying to balance precision and recall.

mean_difference
****************

Returns the ratio of the averages of ``y_pred`` and ``y_true``, with the higher value in the denominator. Assumes
that the values of your outcome column are integers (probably binary).