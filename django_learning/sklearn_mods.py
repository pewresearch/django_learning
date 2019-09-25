from __future__ import print_function
from builtins import str
from builtins import zip
import numbers
import inspect
import time
import warnings
import numpy as np

from traceback import format_exception_only
from abc import ABCMeta, abstractmethod

from sklearn.exceptions import FitFailedWarning
from sklearn.utils._joblib import logger
from sklearn.metrics.scorer import _BaseScorer
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection._validation import _index_param_value
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _num_samples
from sklearn.externals.six import with_metaclass
from sklearn.model_selection._split import BaseCrossValidator


class _ProbaScorer(_BaseScorer):
    def __call__(self, clf, X, y, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.

        Parameters
        ----------
        clf : object
            Trained classifier to use for scoring. Must have a predict_proba
            method; the output of that is used to compute the score.

        X : array-like or sparse matrix
            Test data that will be fed to clf.predict_proba.

        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.

        sample_weight : array-like, optional (default=None)
            Sample weights.

        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """
        y_type = type_of_target(y)
        y_pred = clf.predict_proba(X)
        if y_type == "binary":
            if y_pred.shape[1] == 2:
                y_pred = y_pred[:, 1]
            else:
                raise ValueError(
                    "got predict_proba of shape {},"
                    " but need classifier with two"
                    " classes for {} scoring".format(
                        y_pred.shape, self._score_func.__name__
                    )
                )
        elif len(clf.classes_) > 1 and "pos_label" in list(self._kwargs.keys()):
            for i, label in enumerate(clf.classes_):
                if label == self._kwargs["pos_label"]:
                    y_pred = [p[i] for p in y_pred]
                    break
        if sample_weight is not None:
            return self._sign * self._score_func(
                y, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


def _fit_and_score(
    estimator,
    X,
    y,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    error_score="raise-deprecating",
):
    """Fit estimator and compute scores for a given dataset split.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like, optional, default: None
        The target variable to try to predict in the case of
        supervised learning.

    scorer : A single callable or dict mapping scorer name to the callable
        If it is a single callable, the return value for ``train_scores`` and
        ``test_scores`` is a single float.

        For a dict, it should be one mapping the scorer name to the scorer
        callable object / function.

        The callable object / fn should have signature
        ``scorer(estimator, X, y)``.

    train : array-like, shape (n_train_samples,)
        Indices of training samples.

    test : array-like, shape (n_test_samples,)
        Indices of test samples.

    verbose : integer
        The verbosity level.

    error_score : 'raise' | 'raise-deprecating' or numeric
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If set to 'raise-deprecating', a FutureWarning is printed before the
        error is raised.
        If a numeric value is given, FitFailedWarning is raised. This parameter
        does not affect the refit step, which will always raise the error.
        Default is 'raise-deprecating' but from version 0.22 it will change
        to np.nan.

    parameters : dict or None
        Parameters to be set on the estimator.

    fit_params : dict or None
        Parameters that will be passed to ``estimator.fit``.

    return_train_score : boolean, optional, default: False
        Compute and return score on training set.

    return_parameters : boolean, optional, default: False
        Return parameters that has been used for the estimator.

    return_n_test_samples : boolean, optional, default: False
        Whether to return the ``n_test_samples``

    return_times : boolean, optional, default: False
        Whether to return the fit/score times.

    return_estimator : boolean, optional, default: False
        Whether to return the fitted estimator.

    Returns
    -------
    train_scores : dict of scorer name -> float, optional
        Score on training set (for all the scorers),
        returned only if `return_train_score` is `True`.

    test_scores : dict of scorer name -> float, optional
        Score on testing set (for all the scorers).

    n_test_samples : int
        Number of test samples.

    fit_time : float
        Time spent for fitting in seconds.

    score_time : float
        Time spent for scoring in seconds.

    parameters : dict or None, optional
        The parameters that have been evaluated.

    estimator : estimator object
        The fitted estimator
    """
    # if verbose > 1:
    #     if parameters is None:
    #         msg = ''
    #     else:
    #         msg = '%s' % (', '.join('%s=%s' % (k, v)
    #                       for k, v in parameters.items()))
    #     print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict(
        [(k, _index_param_value(X, v, train)) for k, v in list(fit_params.items())]
    )

    train_scores = {}
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    is_multimetric = not callable(scorer)
    n_scorers = len(list(scorer.keys())) if is_multimetric else 1

    score_params_train = {}
    score_params_test = {}
    try:
        # NOTE: since sample_weight is only passed in fit_params for the training data, we can't pass it on to scorers
        # Also, because different weights may be used for training than for sampling, we don't want to do that anyway
        # Sadly the best way to ensure that models are evaluated using known sampling weights for both train and test
        # Is to directly check for and grab the sampling_weight column from the django learning dataset
        if "sampling_weight" in X.columns:
            score_params_train["sample_weight"] = X_train["sampling_weight"]
            score_params_test["sample_weight"] = X_test["sampling_weight"]
            print("DETECTED DJANGO_LEARNING SAMPLING WEIGHTS, PASSING TO SCORERS")
    except Exception as e:
        pass
        # print("COULDN'T CHECK FOR DJANGO_LEARNING SAMPLING WEIGHTS")
    # if is_multimetric:
    #     score_param_names = []
    #     for name, func in scorer.items():
    #         score_param_names.extend(inspect.getargspec(func._score_func).args)
    #     score_param_names = list(set(score_param_names))
    # else:
    #     score_param_names = inspect.getargspec(scorer._score_func).args
    # score_params_train = {}
    # score_params_test = {}
    # for param in score_param_names:
    #     for k, v in fit_params.items():
    #         if k.endswith(param):
    #             score_params_train[param] = _index_param_value(X, v, train)
    #             # score_params_test[param] = _index_param_value(X, v, test)

    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif error_score == "raise-deprecating":
            warnings.warn(
                "From version 0.22, errors during fit will result "
                "in a cross validation score of NaN by default. Use "
                "error_score='raise' if you want an exception "
                "raised or error_score=np.nan to adopt the "
                "behavior from version 0.22.",
                FutureWarning,
            )
            raise
        elif isinstance(error_score, numbers.Number):
            if is_multimetric:
                test_scores = dict(
                    list(zip(list(scorer.keys()), [error_score] * n_scorers))
                )
                if return_train_score:
                    train_scores = dict(
                        list(zip(list(scorer.keys()), [error_score] * n_scorers))
                    )
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
            warnings.warn(
                "Estimator fit failed. The score on this train-test"
                " partition for these parameters will be set to %f. "
                "Details: \n%s" % (error_score, format_exception_only(type(e), e)[0]),
                FitFailedWarning,
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a"
                " numeric value. (Hint: if using 'raise', please"
                " make sure that it has been spelled correctly.)"
            )

    else:
        fit_time = time.time() - start_time
        # _score will return dict if is_multimetric is True
        test_scores = _score(
            estimator, X_test, y_test, scorer, is_multimetric, **score_params_test
        )
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(
                estimator,
                X_train,
                y_train,
                scorer,
                is_multimetric,
                **score_params_train
            )

    # if verbose > 2:
    #     if is_multimetric:
    #         for scorer_name, score in test_scores.items():
    #             msg += ", %s=%s" % (scorer_name, score)
    #     else:
    #         msg += ", score=%s" % test_scores
    # if verbose > 1:
    #     total_time = score_time + fit_time
    #     end_msg = "%s, total=%s" % (msg, logger.short_format_time(total_time))
    #     print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_scores, test_scores] if return_train_score else [test_scores]

    if return_n_test_samples:
        ret.append(_num_samples(X_test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_estimator:
        ret.append(estimator)
    return ret


def _score(estimator, X_test, y_test, scorer, is_multimetric=False, **score_params):
    """Compute the score(s) of an estimator on a given test set.

    Will return a single float if is_multimetric is False and a dict of floats,
    if is_multimetric is True
    """
    if is_multimetric:
        return _multimetric_score(estimator, X_test, y_test, scorer, **score_params)
    else:
        if y_test is None:
            score = scorer(estimator, X_test, **score_params)
        else:
            score = scorer(estimator, X_test, y_test, **score_params)

        if hasattr(score, "item"):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass

        if not isinstance(score, numbers.Number):
            raise ValueError(
                "scoring must return a number, got %s (%s) "
                "instead. (scorer=%r)" % (str(score), type(score), scorer)
            )
    return score


def _multimetric_score(estimator, X_test, y_test, scorers, **score_params):
    """Return a dict of score for multimetric scoring"""
    scores = {}

    for name, scorer in list(scorers.items()):

        score_param_names = inspect.getargspec(scorer._score_func).args
        scorer_params = {
            k: v for k, v in list(score_params.items()) if k in score_param_names
        }
        if y_test is None:
            score = scorer(estimator, X_test, **scorer_params)
        else:
            score = scorer(estimator, X_test, y_test, **scorer_params)

        if hasattr(score, "item"):
            try:
                # e.g. unwrap memmapped scalars
                score = score.item()
            except ValueError:
                # non-scalar?
                pass
        scores[name] = score

        if not isinstance(score, numbers.Number):
            raise ValueError(
                "scoring must return a number, got %s (%s) "
                "instead. (scorer=%s)" % (str(score), type(score), name)
            )
    return scores
