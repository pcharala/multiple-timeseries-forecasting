# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:00:59 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""
from common.Utils import Utils
from common import model_desc as md

# import models


class Trainer(object):
    """
    A class that represents a trainer.
    
    Attributes
    ----------
    dp: DataPreparation
        a DataPreparation object.
    model: dict
        a dictionary holding the model attributes and parameters.
    error_metrics: list[str]
        a list with error metrics to be computed.
    """
    
    def __init__(self,
                 dp,
                 model,
                 error_metrics=['smape']):
        
        """
        Example:
        model = {
            'name':'BHTArimaModel',
            'params': {
                'p':1,
                'd':1,
                'q':1,
                'tau':2,
                'Rs':[3,3],
                'k':10,
                'tol':0.001,
                'Us_mode':4
                },
            'metrics': ['rmse', 'mse', 'mape']
            }
        """
        
        self._dp = dp
        self._model = model
        self._error_metrics = error_metrics


    def train_and_predict(self):
        """
        Perform training and prediction.

        Returns
        -------
        None.

        """
        result_dict = dict()
        result_dict['metrics'] = {key: None 
                                   for key in self._error_metrics}
        result_dict['params'] = self._model['params']
        
        model_desc = md.generate_model_description(self._model['name'],
                                                   self._model['params'])
        
        # load model class and instantiate
        model_mod = getattr(__import__('models.' +
                                       self._model['name']),
                              self._model['name'])
        model_class = getattr(model_mod, self._model['name'])
        model = model_class(model_desc)
        
        print('---------------------------------------------------')
        print('Trainer hyperparameter set {}'.format(self._model['params']))
        
        try:
            self._dp.reset_lookback_history(
                self._model['params']['lookback_history'])
        except KeyError:
            pass
        
        # select model
        if self._model['name'] in ['BHTArimaModel']:
            
            insample = self._dp.get_insample()
            outsample = self._dp.get_outsample()
            
            model.fit(insample)
            prediction = model.predict(outsample)
        
        elif self._model['name'] in ['LSTMModel',
                                     'CNNModel',
                                     'BiLSTMModel',
                                     'CNNLSTMModel']:
            
            (insample_X,
             insample_y,
             outsample_X,
             outsample_y) = self._dp.get_supervised_outsample()
                        
            model.fit((insample_X, insample_y))
            prediction = model.predict((outsample_X, outsample_y)).transpose()
            outsample = outsample_y.transpose()
            
        elif self._model['name'] in ['DeepESNModel']:

            (insample_X,
             insample_y,
             outsample_X,
             outsample) = self._dp.get_supervised_outsample(list_form=True)
        
            model.fit((insample_X, insample_y, outsample_X))
            prediction = model.predict((outsample_X, outsample)).transpose()
            outsample = outsample.transpose()
        
        elif self._model['name'] in ['RFModel',
                                     'SVRModel']:
            
            # According to the latest proposed CV scheme,
            # each fold uses a 'lookback history' of len(train_samples)
            self._dp.reset_lookback_history()
            
            try:
                (insample_X,
                 insample_y,
                 outsample_X,
                 outsample) = self._dp.get_outsample(xy_form=True)
                
                model.fit((insample_X, insample_y))
                prediction = model.predict((outsample_X, outsample))
            except StopIteration:
                self._dp.reset_wfcv_gen()
        
        else:
            raise ValueError('Unknown model name')


        # inverse scaling
        prediction = self._dp.inverse_scaling(prediction)
        outsample = self._dp.inverse_scaling(outsample)
        
        # compute errors
        for metric in result_dict['metrics']:
            result_dict['metrics'][metric] = \
                Utils.compute_pred_error(outsample, 
                                         prediction,
                                         metric.lower()
                                         )
                
        print('Outsample prediction metrics: {}'.format(
            result_dict['metrics']))
        print('---------------------------------------------------')
                
        return result_dict