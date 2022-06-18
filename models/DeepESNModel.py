# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 12:25:31 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

from .BaseModel import BaseModel

import numpy as np
from DeepESN import DeepESN


class DeepESNModel(BaseModel):
    """
    A class that represents a Deep ESN model.
    
    Attributes
    ----------
    params: dict
        A dictionary that holds model parameters.
    """

    def __init__(self, params):
        super().__init__(params)
        self._trained_model = None
    
    
    def _validate_params(self):
        pass


    @staticmethod
    def _select_indices(data, indexes, transient=0):

        if len(data) == 1:
            return [data[0][:,indexes][:,transient:]]
        
        return [data[i][:,transient:] for i in indexes]
    
    
    def fit(self, train_data):
        
        # We also need test data input for initializing DeepESN model states.
        # This is an intrinsic particularity of DeepESN API.
        (train_X, train_Y, test_X) = train_data
        
        Nu = train_X.shape[2] # number of streams
        tr_idxs = list(range(train_X.shape[0]))

        train_X = [x.transpose() for x in train_X]
        train_Y = [x.transpose() for x in train_Y]
        test_X = [x.transpose() for x in test_X]
                
        # create config struct from dictionary
        configs = Struct(**{x: self._params[x] for x in self._params
                   if x not in ['Nr', 'Nl', 'reg', 'transient']})
        configs.IPconf.indexes = tr_idxs
        
        self._trained_model = DeepESN(Nu,
                              self._params['Nr'],
                              self._params['Nl'],
                              configs,
                              verbose=0)

        self._states = \
            self._trained_model.computeState(train_X + test_X,
                                             self._trained_model.IPconf.DeepIP)
        
        # print('ESN states [train + test] len: ', len(self._states))

        train_states = DeepESNModel._select_indices(self._states,
                                                    tr_idxs,
                                                    self._params['transient'])
        
        # print('Train states len: ', len(train_states))
        
        train_targets = DeepESNModel._select_indices(train_Y,
                                                     tr_idxs,
                                                     self._params['transient'])
        
        # print('Train targets len: ', len(train_targets))

        self._trained_model.trainReadout(train_states,
                                         train_targets,
                                         self._params['reg'])
    
    
    def predict(self, test_data):
        
        if self._trained_model:

            (test_X, test_Y) = test_data
            
            timesteps = test_X.shape[1]
            
            test_X = [x.transpose() for x in test_X]
            test_Y = [x.transpose() for x in test_Y]
            
            tr_len = len(self._states) - len(test_X)
            st_te_idxs = list(range(tr_len, len(self._states)))
            
            test_states = DeepESNModel._select_indices(self._states,
                                                       st_te_idxs)
            
            # print('Test states len: ', len(test_states))

            test_outputs = self._trained_model.computeOutput(test_states)
            
            # print('Test_outputs len: ', len(test_states))
            
            prediction = list()
            for te_idx in range(len(test_Y)):
                prediction.append(test_outputs[:,
                                               (te_idx+1)*timesteps-1])
                
            prediction = np.array(prediction)
            
            # print('Prediction shape: ', prediction.shape)

            return prediction
        raise ValueError('Model is not trained')


class Struct(object):
    def __init__(self, **entries):
        for k,v in entries.items():
            if isinstance(v,dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v