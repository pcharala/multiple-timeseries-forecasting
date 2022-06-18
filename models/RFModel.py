# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:41:37 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

from .BaseModel import BaseModel

import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RFModel(BaseModel):
    """
    A class that represents an RF model.
    
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
        
        (train_X, train_y) = train_data
        
        model = \
            RandomForestRegressor(n_estimators=self._params['n_estimators'],
                                  max_features=self._params['max_features'],
                                  max_depth=self._params['max_depth'],
                                  min_samples_split=self._params[
                                      'min_samples_split'],
                                  min_samples_leaf=self._params[
                                      'min_samples_leaf'],
                                  bootstrap=self._params['bootstrap'],
                                  n_jobs=-1,
                                  verbose=0)
                
        model.fit(train_X, train_y)   
        self._trained_model = model


    def predict(self, test_data):
        
        (test_X, test_y) = test_data
        
        if self._trained_model:
            return np.expand_dims(self._trained_model.predict(test_X), axis=1)
        raise ValueError('Model is not trained')
