# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 01:29:13 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

from .BaseModel import BaseModel

import numpy as np
from BHT_ARIMA import BHTARIMA

class BHTArimaModel(BaseModel):
    """
    A class that represents a BHT-ARIMA model.
    
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
    
    
    def fit(self, train_data):

        model = BHTARIMA(train_data,
                         self._params['p'],
                         self._params['d'],
                         self._params['q'],
                         [train_data.shape[0], self._params['tau']],
                         [self._params['tau'], self._params['tau']],
                         self._params['k'],
                         self._params['tol'],
                         Us_mode=self._params['Us_mode'],
                         verbose=0)
        self._trained_model, _ = model.run()
    
    
    def predict(self, test_data):
        if self._trained_model.any():
            prediction = np.expand_dims(self._trained_model[..., -1],
                                        axis=1)
            return prediction
        raise ValueError('Model is not trained')
