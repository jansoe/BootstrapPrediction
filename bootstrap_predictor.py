import logging
import numpy as np
from numpy.random import random_integers
from collections import defaultdict
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

class BootstrapRegressor():
    ''' (Stratified) Bootstrap Wrapper for Regressors

    The regressor is trained for different bootstrap samples of the data.
    The data is split at strat_thres and individual bootstrap samples are drawn
    from both sets.

    Parameter
    ---------
    regressor: a regressor class in sklearn syntax

    reg_param: a dictionary of regressor parameter

    n_member: number of ensemble members (default 50)

    strat_thres: point at which data is stratified

    aggregate: function to combine individual regressor predictions
               (default np.mean)

    Attributes
    ----------
    full_regressor: regressor trained on the all available data (wto bootstrap)

    regressors: list of regressors trained on bootstrap data

    _pred_dict: dictionary of target id to list of predictions of regressors
                not trained on the targets

    _target_dict: dictionary of target id to target value

    oob_prediction: aggregated prediction of regressors not trained on samples

    oob_score_: r2 score of aggregated prediction

    oob_score_single: r2 score of individual predictions

    Methods
    -------

    fit: fit regressors to targets

    predict: bagged prediction for features
    '''

    def __init__(self, regressor, reg_param, n_member=50, strat_thres=0.5,
                 aggregate=np.mean):
        self.full_regressor = regressor(**reg_param)
        self.regressors = [regressor(**reg_param) for i in range(n_member)]
        self.thres = strat_thres
        self._pred_dict = defaultdict(list)
        self._train_targets = {}
        self.aggregate=aggregate

    def fit(self, features, targets):
        ''' fit the regressors to targets'''

        self.full_regressor.fit(features, targets)
        for reg_ind, regressor in enumerate(self.regressors):
            train_idx = _strat_bootstrap(targets, self.thres)
            test_idx = list(set(range(len(targets))).difference(train_idx))
            regressor.fit(features[train_idx], targets[train_idx])
            prediction = regressor.predict(features[test_idx])
            for tar, pred in zip(test_idx, prediction):
                self._pred_dict[tar].append(pred)
        self._train_targets = dict(zip(range(len(targets)), targets))

    def predict(self, features):
        ''' aggregated bootstrap prediction'''

        predictions = []
        for reg_ind in range(1, len(self.regressors)):
            predictions.append(self.regressors[reg_ind].predict(features))
        return self.aggregate(np.vstack(predictions), 0)

    def _oob_prediction_single(self):
        ''' returns list of individual predictions and list of target values'''

        tar_list, pred_list = [], []
        for tar_id, preds in self._pred_dict.items():
            tar_list += [self._train_targets[tar_id]] * len(preds)
            pred_list += preds
        return pred_list, tar_list

    @property
    def oob_prediction(self):
        return np.array([self.aggregate(self._pred_dict[i])
                         for i in self._train_targets])

    @property
    def oob_score_(self):
        mask = np.isnan(self.oob_prediction)
        targets = np.array(self._train_targets.values())
        prediction = self.oob_prediction
        if np.sum(mask) > 0:
            logging.warning('Not all values bootstrap predicted')
            targets = targets[~mask]
            prediction = prediction[~mask]
        return r2_score(targets, prediction)

    @property
    def oob_score_single(self):
        pred, tar = self._oob_prediction_single()
        return r2_score(tar, pred)


def _strat_bootstrap(targets, strat_thres=0.5):
    ''' performs stratified bootstrap sampling'''

    division = targets >= strat_thres
    ind1 = np.where(division)[0]
    ind2 = np.where(~division)[0]
    if len(ind1 > 1):  # data is seperated by strat_thres
        bootstr = np.hstack([ind1[random_integers(0, len(ind1)-1, len(ind1))],
                             ind2[random_integers(0, len(ind2)-1, len(ind2))]])
    else:
        # no stratification
        num = len(targets)
        bootstr = random_integers(0, num-1, num)
    return bootstr


class FilteredRegressor():
    ''' Class for wrapping a filter around a regressor'''

    def __init__(self, filtering, regressor, filt_param={}, reg_param={}):
        self.filtering = filtering(**filt_param)
        self.regressor = regressor(**reg_param)

    def fit(self, features, targets):
        fprep = self.filtering.fit_transform(features, targets)
        self.regressor.fit(fprep, targets)

    def predict(self, features):
        fprep = self.filtering.transform(features)
        return self.regressor.predict(fprep)

class ForestFilter():
    ''' Random Forest Scoring Function for sklearn feature selection'''

    def __init__(self, **param):
        self.forest = RandomForestRegressor(**param)

    def __call__(self, features, targets):
        self.forest.fit(features, targets)
        return self.forest.feature_importances_, None
