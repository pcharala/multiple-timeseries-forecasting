# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 01:31:06 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

from abc import ABC, abstractmethod

class BaseModel(ABC):
    
    def __init__(self, params):
        self._params = params
        

    def get_params(self):
        return self._params
    

    @abstractmethod
    def _validate_params(self):
        pass
    
    
    @abstractmethod
    def fit(self, train_data):
        pass
    
    
    @abstractmethod
    def predict(self, test_data):
        pass
