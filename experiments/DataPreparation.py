# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:31:58 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

import numpy as np
#from numpy.lib.stride_tricks import sliding_window_view
from tscv import GapRollForward
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import FunctionTransformer


class DataPreparation(object):
    """
    A class that represents a data-preparation object.
    
    Attributes
    ----------
    ts: ndarray
        a numpy array holding timeseries under processing.
        Rows correspond to different timeseries and columns to time steps.
    perc_outsample: int
        the percentage of observations number to keep as out-of-sample.
    len_outsample: int
        the number of observations to keep as out-of-sample.
    perc_valid: int
        the percentage of "in-sample" observations kept as validation set.
    fh: int
        the forecasting horizon.
    lookback_history: int
        the size of the rolling window or the min-size of the expanding window.
    window_type: str
        the type of the window. Valid values are 'rolling' and 'expanding'.
    scaler: str
        the type of scaler used. Valid values are 'minmax', 'standard' and
        'log1p'.
    verbose: int
        verbose output
    """
    
    def __init__(self,
                 ts,
                 perc_outsample=None,
                 len_outsample=None,
                 perc_valid=20,
                 fh=1,
                 lookback_history=None,
                 window_type='rolling',
                 scaler=None,
                 verbose=0):
    
        self._ts = ts
        ts_nsen, ts_nobs = ts.shape
        self._wfcv_gapsize = 0
        self._wfcv_maxtestsize = fh
        self._wfcv_mintestsize = fh
        self._window_type = window_type
        self._scaler = self._init_scaler(scaler)
        self._verbose = verbose
        
        if perc_outsample:
            self._len_outsample = int(ts_nobs*perc_outsample//100)
        elif len_outsample:
            self._len_outsample = len_outsample
        else:
            raise ValueError("Provide either percentage "
                             "or length of out-of-sample")
            
        self._insample = ts[:, :-self._len_outsample]
        self._outsample = ts[:, -self._len_outsample:]
        
        # necessary for supervised learning methods
        self._val_size = int(self._insample.shape[1] * perc_valid // 100)
        self._tr_size = self._insample.shape[1] - self._val_size 

        if self._verbose:
            print('---------------------------------------------------')
            print('Insample length: {}'.format(self._insample.shape[1]))
            print('Validation length: {}'.format(self._val_size))
            print('Outsample length: {}'.format(self._outsample.shape[1]))
            print('---------------------------------------------------')        

        if lookback_history:
            self._lookback_history = lookback_history
        else:
            self._lookback_history = self._insample.shape[1] - self._val_size        

        # fit scaler on insample
        if self._scaler:
            self._scaler.fit(np.transpose(self._insample))
        
        #_min_train_size = self._insample.shape[1]*perc_train//100
        # print('Min train size: {}'.format(_min_train_size))
        
        self._init_wfcv(self._window_type,
                        self._wfcv_gapsize,
                        self._wfcv_maxtestsize,
                        self._wfcv_mintestsize) 

        self._cached_folds = self._cache_folds_insample()


    def _init_scaler(self, scaler):
        """
        Initialize scaler.

        """
        if not scaler:
            return None
        elif scaler == 'minmax':
            return MinMaxScaler(feature_range = (0, 1))
        elif scaler == 'standard':
            return StandardScaler()
        elif scaler == 'log1p':
            return FunctionTransformer(np.log1p, inverse_func = np.expm1)
        else:
            raise ValueError('Unknown scaler type')


    def _init_wfcv(self,
                   window_type,
                   wfcv_gapsize,
                   wfcv_maxtestsize,
                   wfcv_mintestsize):
        """
        Initialize walk-forward CV.

        """
        if window_type == 'rolling':
            self._wfcv = GapRollForward(min_train_size=self._lookback_history,
                                        max_train_size=self._lookback_history,
                                        gap_size=wfcv_gapsize,
                                        max_test_size=wfcv_maxtestsize,
                                        min_test_size=wfcv_mintestsize)
        elif window_type == 'expanding':
            self._wfcv = GapRollForward(min_train_size=self._lookback_history,
                                        gap_size=wfcv_gapsize,
                                        max_test_size=wfcv_maxtestsize,
                                        min_test_size=wfcv_mintestsize)
        else:
            raise ValueError('Unknown window type')
        
        if self._verbose:            
            print('Resetting folds')
            
        self.reset_wfcv_gen()


    def reset_lookback_history(self, lookback_history=None):
        """
        Reset lookback history.
        
        Parameters
        ----------
        lookback_history: int
            the size of the rolling window or
            the min-size of the expanding window.

        Returns
        -------
        None.

        """
        if lookback_history:
            self._lookback_history = lookback_history
        else:
            self._lookback_history = self._insample.shape[1] - self._val_size

        if self._verbose:
            print('New lookback history: {}'.format(lookback_history))
            
        self._init_wfcv(self._window_type,
                        self._wfcv_gapsize,
                        self._wfcv_maxtestsize,
                        self._wfcv_mintestsize)

    
    def get_training_size(self):
        return self._tr_size


    def get_insample(self, scaled=True):
        if self._scaler and scaled:
            return np.transpose(
                self._scaler.transform(
                    np.transpose(self._insample)))
        else:
            return self._insample

    
    def get_outsample(self,
                      scaled=True,
                      xy_form=False):
        
        if self._scaler and scaled:
            outsample =  np.transpose(
                            self._scaler.transform(
                                np.transpose(self._outsample)))
        else:
            outsample = self._outsample
        
        if xy_form:
            return (self.get_insample(scaled)[:, :-1],
                    self.get_insample(scaled)[:, -1],
                    self.get_insample(scaled)[:, 1:],
                    outsample)
        else:
            return outsample
    
    
    def get_ts(self, scaled=True):
        if self._scaler and scaled:
            return np.transpose(
                self._scaler.transform(
                    np.transpose(self._ts)))
        else:
            return self._ts
    
    
    def get_wfcv(self):
        return self._wfcv


    def get_next_fold_insample(self,
                               scaled=True,
                               xy_form=False):
        tr_idx, te_idx = next(self._split_gen)
        if xy_form:
            if self._verbose:
                print('Selected indices [train_x] [train_y] '
                      '[test_x] [test_y]: ',
                      tr_idx[:-1], tr_idx[-1], tr_idx[1:], te_idx)
            return (self.get_insample(scaled)[:, tr_idx[:-1]],
                    self.get_insample(scaled)[:, tr_idx[-1]],
                    self.get_insample(scaled)[:, tr_idx[1:]],
                    self.get_insample(scaled)[:, te_idx])
        else:
            if self._verbose:
                print('Selected indices [train] [test]: ', tr_idx, te_idx)
            return (self.get_insample(scaled)[:, tr_idx],
                    self.get_insample(scaled)[:, te_idx])
    
    
    def _cache_folds_insample(self, scaled=True):
        
        list_of_splits = list()
        
        try:
            while True:
                tr_idx, te_idx = next(self._split_gen)
                list_of_splits.append(
                        (self.get_insample(scaled)[:, tr_idx],
                         self.get_insample(scaled)[:, te_idx]))   
        except StopIteration:
            self.reset_wfcv_gen()
        
        if self._verbose:
            print('Cached Generator.')
            
        return list_of_splits
    
    
    def get_cached_folds(self):
        return self._cached_folds
    
    
    def get_scaler(self):
        return self._scaler
    

    def _to_supervised(self,
                       ts,
                       list_form=False):
        X, y = list(), list()
        for i in range(len(ts)):
            # find the end of this pattern
            end_ix = i + self._lookback_history
            # check if we are beyond the dataset
            if end_ix > len(ts)-1:
                break
            # gather input and output parts of the pattern
            if list_form:
                (seq_x, seq_y) = (ts[i:end_ix, :],
                                  ts[i+1:end_ix+1, :])
            else:
                (seq_x, seq_y) = (ts[i:end_ix, :],
                                  ts[end_ix, :])
            X.append(seq_x)
            y.append(seq_y)
        
        array_X = np.array(X)
        array_y = np.array(y)
        
        return (array_X, array_y)

    
    def get_supervised_insample(self,
                                scaled=True,
                                list_form=False):
        # prepare insample data for supervised learning.
        # last window is used as test set
        array_X, array_y = \
            self._to_supervised(self.get_insample(scaled).transpose(),
                                list_form)
        
        if list_form:
            return (array_X[:-self._val_size, :, :],
                    array_y[:-self._val_size, :, :],
                    array_X[-self._val_size:, :, :],
                    np.squeeze(array_y[-self._val_size:, -1:, :], axis=1))
        else:
            return (array_X[:-self._val_size, :, :],
                    array_y[:-self._val_size, :],
                    array_X[-self._val_size:, :, :],
                    array_y[-self._val_size:, :])
        

    def get_supervised_outsample(self,
                                 scaled=True,
                                 list_form=False):
        # prepare entire dataset for supervised learning.
        # last window is used as test set
        array_X, array_y = \
            self._to_supervised(self.get_ts(scaled).transpose(),
                                list_form)
        
        # X array dims: [num_windows, win_size, num_streams]
        if list_form:
            return (array_X[:-1, :, :],
                    array_y[:-1, :, :],
                    array_X[-1:, :, :],
                    np.squeeze(array_y[-1:, -1:, :], axis=1))
        else:
            return (array_X[:-1, :, :],
                    array_y[:-1, :],
                    array_X[-1:, :, :],
                    array_y[-1:, :])
  
    
    def inverse_scaling(self, inp):
        if self._scaler:
            return np.transpose(
                self._scaler.inverse_transform(
                    inp.transpose()))
        else:
             return inp
        
        
    def reset_wfcv_gen(self):
        self._split_gen = self._wfcv.split(self._insample.transpose())
        if self._verbose:
            print('Total number of folds: {}'
                  .format(len(list(self._split_gen))))
        self._split_gen = self._wfcv.split(self._insample.transpose())
