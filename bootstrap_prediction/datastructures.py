import logging, copy
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from functools import reduce

class FeatureBank():
    ''' A simple class to store feature array with metadata

        Parameters
        ----------
        feat_name: iterable of feature names

        obj_name: iterable of sample names

        data: np.array

        Methods
        -------
        zscore: zscores the data
    '''

    def __init__(self, feat_name, obj_name, data):
        self.feat_name = list(feat_name)
        self.obj_name = list(obj_name)
        self.data = data

    def zscore(self):
        ''' zscores the data'''
        self.scaler = preprocessing.StandardScaler()
        self.data = self.scaler.fit_transform(self.data)


def combine_features(feature_bank_list):
    ''' combines a list of FeatureBank instances to a single Feature Bank

        selects common objects of all feature banks and appends their feature
    '''

    common_obj = list(reduce(lambda x, y: set(x).intersection(y),
                            [f.obj_name for f in feature_bank_list]))
    combined_feature, combined_feat_name = [], []
    for f_bank in feature_bank_list:
        sel_obj = [f_bank.obj_name.index(i) for i in common_obj]
        combined_feature.append(f_bank.data[sel_obj])
        combined_feat_name += copy.copy(f_bank.feat_name)
    new_f_bank = FeatureBank(combined_feat_name, common_obj,
                             np.hstack(combined_feature))
    return new_f_bank


class TrainData():
    ''' A simple class to handle feature and target values

        Parameters
        ----------
        target_dict: dictionary of target_id to response

        feature_bank: FeatureBank instance

        Attributes
        ----------
        targets: array of target values

        features: array of feature values

        obj_name: list of sample names

        feat_name: list of feature names
    '''

    def __init__(self, target_dict, feature_bank):

        # obtain available features for targets
        obj_wt_feat = list(feature_bank.obj_name)
        tar_wt_feat = list(set(target_dict.keys()).intersection(obj_wt_feat))
        which_feature = [obj_wt_feat.index(i) for i in tar_wt_feat]
        logging.info('No features available for: {mol!s}'.format(
                    mol = set(target_dict.keys()).difference(tar_wt_feat)))
        self.targets = np.array([target_dict[i] for i in tar_wt_feat])
        self.features = feature_bank.data[which_feature].copy()
        self.obj_name = tar_wt_feat
        self.feat_name = copy.copy(feature_bank.feat_name)

    def feature_selection(self, mask):
        ''' select feature according to boolean mask'''
        self.features = self.features[:, mask]
        if self.feat_name:
            self.feat_name = [self.feat_name[i] for i in np.where(mask)[0]]

    def y_randomization(self):
        ''' randomly permute targets'''
        permutation = np.random.permutation(range(len(self.targets)))
        self.targets = self.targets[permutation]
        self.obj_name = [self.obj_name[i] for i in permutation]
