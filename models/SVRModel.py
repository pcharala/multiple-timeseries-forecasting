# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:50:37 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

from .BaseModel import BaseModel

import numpy as np
from sklearn.svm import SVR

class SVRModel(BaseModel):
    """
    A class that represents an SVR model.
    
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
        
        model = SVR(kernel=self._params['kernel'],
                    C=self._params['C'],
                    gamma=self._params['gamma'])
                
        model.fit(train_X, train_y)   
        self._trained_model = model


    def predict(self, test_data):
        
        (test_X, test_y) = test_data
        
        if self._trained_model:
            return np.expand_dims(self._trained_model.predict(test_X), axis=1)
        raise ValueError('Model is not trained')
