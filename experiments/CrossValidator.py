# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:31:58 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

from numpy.random import permutation

from common.Utils import Utils
from common import model_desc as md
from sklearn.model_selection import ParameterGrid
from collections import Counter

import os
import math
from joblib import Parallel, delayed

#import models

import tensorflow as tf
import gc

class CrossValidator(object):
    """
    A class that represents a cross-validator.
    
    Attributes
    ----------
    dp: DataPreparation
        a DataPreparation object.
    hypermodel: dict
        a dictionary holding the hyper-model attributes and hyperparameters.
    error_metrics: list[str]
        a list with error metrics to be computed.
    decision_method: str
        the name of error metric used during cross-validation.
        Alternatively, 'majority_voting' for choosing best hyperparams
    search_type: str
        the name of hyperparameter search type. Valid values are 'grid' and
        'random'.
    search_max_num_points: int
        the max number of search points, in case of random search for
        hyperparameters.
    ncpus: int
        the number of cpus for CV multiprocessing. If value is -1, then use all
        available system cpus.
    """
    
    def __init__(self,
                 dp,
                 hypermodel,
                 error_metrics=['smape'],
                 decision_method='majority_voting',
                 search_type='grid',
                 search_max_num_points=100,
                 ncpus=-1):
        
        self._dp = dp
        self._hypermodel = hypermodel
        self._error_metrics = error_metrics
        self._decision_method = decision_method
        self._search_type = search_type
        self._search_max_num_points = search_max_num_points
        self._best_model_hparams = None
                
        if ncpus > os.cpu_count() or ncpus==-1:
            ncpus = os.cpu_count()
        
        self._executor = Parallel(n_jobs=ncpus, backend='multiprocessing')

    
    @staticmethod
    def _process_combination(comb, hname, cached_folds, scaler):
        """
        Process current hyperparameters combination.

        """
        
        print('Current hyperparameter set {}'.format(comb['params']))
        
        model_desc = \
            md.generate_model_description(hname,
                                          comb['params'])
            
        # load model class and instantiate
        model_mod = getattr(__import__('models.' + hname), hname)
        model_class = getattr(model_mod, hname)
        model = model_class(model_desc)
        
        for fold in cached_folds:
            
            train, valid = fold
            
            model.fit(train)
            prediction = model.predict(valid)
            
            if scaler:
                prediction = scaler.inverse_transform(
                    prediction.transpose()).transpose()
                valid = scaler.inverse_transform(
                    valid.transpose()).transpose()
            
            # compute errors
            for metric in comb['metrics']:
                comb['metrics'][metric].append(
                    Utils.compute_pred_error(valid, 
                                              prediction,
                                              metric.lower()
                                              )
                    )
                
        return comb


    def run(self):
        """
        Run the cross-validator.

        Returns
        -------
        None.

        """
        
        if self._decision_method \
            not in self._error_metrics:
            if self._decision_method != 'majority_voting':
                raise ValueError('Invalid decision method')
        
        # compute absolute values from relative ones for number of inputs
        try:
            n_inputs = [math.ceil(x * self._dp.get_training_size())
                             for x in 
                        self._hypermodel['params']['lookback_history']]
            
            self._hypermodel['params']['lookback_history'] = n_inputs
            
        except:
            pass
          
        comb_list = []

        if self._search_type == 'grid':
            # generate all combination of hyperparameters
            hyper_grid = list(ParameterGrid(self._hypermodel['params']))
        elif self._search_type == 'random':
            # generate all combination of hyperparameters
            hyper_grid = list(ParameterGrid(self._hypermodel['params']))
            idx = permutation(len(hyper_grid))[:self._search_max_num_points]
            hyper_grid = [hyper_grid[i] for i in idx]
        else:
            raise ValueError('Unknown hyperparameters search type')
            
        for hyper_point in hyper_grid:
            
            ###### compute absolute values from relative ones
            try:
                hyper_point['batch_size'] = math.ceil(
                    hyper_point['batch_size'] *
                    hyper_point['lookback_history'])
            except:
                pass
            
            ###### compute absolute values from relative ones
            
            metrics_dict = dict()
            # initialize metrics lists
            metrics_dict['metrics'] = {key: list() 
                                       for key in self._error_metrics}
            metrics_dict['params'] = hyper_point
            comb_list.append(metrics_dict)
        
        try:
            # if lookback_history, sort by
            comb_list = \
                sorted(comb_list,
                       key=lambda d: d['params']['lookback_history'])
        except:
            pass

        # init average metrics:
        avg_metrics = {key: list() 
                           for key in self._error_metrics}
        
        total_comb_num = len(comb_list)
        
        print('---------------------------------------------------')
        
        if self._hypermodel['name'] in ['BHTArimaModel']:
            
            tasks = (delayed(CrossValidator._process_combination)(
                comb,
                self._hypermodel['name'],
                self._dp.get_cached_folds(),
                self._dp.get_scaler()) for comb in comb_list)
            
            tasks_out = self._executor(tasks)
                        
            for comb in tasks_out:    
                # ------------------------------------------------------------
                for metric in comb['metrics']:
                    value = sum(comb['metrics'][metric])/\
                                 len(comb['metrics'][metric])
                    print("{}: {}".format(metric, value))
                    avg_metrics[metric].append(value)    
                print('---------------------------------------------------')

        else:
            # select model
            for idx, comb in enumerate(comb_list):
                print('Grid point {}/{}'.format(idx+1, total_comb_num))
                print('Current hyperparameter set {}'.format(comb['params']))
                
                try:
                    self._dp.reset_lookback_history(
                        comb['params']['lookback_history'])
                except KeyError:
                    pass

                model_desc = \
                    md.generate_model_description(
                        self._hypermodel['name'],
                        comb['params'])
                    
                # load model class and instantiate
                model_mod = getattr(__import__('models.' +
                                                 self._hypermodel['name']),
                                      self._hypermodel['name'])
                model_class = getattr(model_mod, self._hypermodel['name'])
                model = model_class(model_desc)

                if self._hypermodel['name'] in ['LSTMModel',
                                                'CNNModel',
                                                'BiLSTMModel',
                                                'CNNLSTMModel']:
    
                    (train_X, train_y, valid_X, valid_y) = \
                        self._dp.get_supervised_insample()

                    model.fit((train_X, train_y))
                    prediction = model.predict((valid_X, valid_y))
                                        
                    # inverse scaling
                    prediction = \
                        self._dp.inverse_scaling(prediction.transpose())
                    valid_y = self._dp.inverse_scaling(valid_y.transpose())
                    
                    for i in range(valid_y.shape[1]):
                        # compute errors
                        for metric in comb['metrics']:
                            comb['metrics'][metric].append(
                                Utils.compute_pred_error(valid_y[:,i], 
                                                         prediction[:,i],
                                                         metric.lower()
                                                         )
                                )
                    tf.keras.backend.clear_session()
                    del model
                    gc.collect()
                            
                elif self._hypermodel['name'] in ['DeepESNModel']:
                    
                    (train_X, train_y, valid_X, valid_y) = \
                        self._dp.get_supervised_insample(list_form=True)
                        
                    model.fit((train_X, train_y, valid_X))
                    prediction = model.predict((valid_X, valid_y))
                                        
                    # inverse scaling
                    prediction = \
                        self._dp.inverse_scaling(prediction.transpose())
                    valid_y = self._dp.inverse_scaling(valid_y.transpose())
                    
                    for i in range(valid_y.shape[1]):
                        # compute errors
                        for metric in comb['metrics']:
                            comb['metrics'][metric].append(
                                Utils.compute_pred_error(valid_y[:,i], 
                                                          prediction[:,i],
                                                          metric.lower()
                                                          )
                                )
                            
                elif self._hypermodel['name'] in ['RFModel',
                                                  'SVRModel']:

                    self._dp.reset_lookback_history()
                    
                    try:
                        while True:
                            # Parameters setting
                            (train_X, train_y, valid_X, valid_y) = \
                                self._dp.get_next_fold_insample(xy_form=True)
                            # print(train.shape, valid.shape)
    
                            model.fit((train_X, train_y))
                            prediction = model.predict((valid_X, valid_y))
                            
                            # inverse scaling
                            prediction = self._dp.inverse_scaling(prediction)
                            valid = self._dp.inverse_scaling(valid_y)
                            
                            # compute errors
                            for metric in comb['metrics']:
                                comb['metrics'][metric].append(
                                    Utils.compute_pred_error(valid, 
                                                             prediction,
                                                             metric.lower()
                                                             )
                                    )
                    except StopIteration:
                        self._dp.reset_wfcv_gen()
                        
                else:
                    raise ValueError('Unknown model name ' 
                                     + self._hypermodel['name'])
                              
               
                # -------------------------------------------------------------
                for metric in comb['metrics']:
                    value = sum(comb['metrics'][metric])/\
                                 len(comb['metrics'][metric])
                    print("{}: {}".format(metric, value))
                    avg_metrics[metric].append(value)    
                print('---------------------------------------------------')                

        # select best hyperparameters combination
        if self._decision_method == 'majority_voting':
            votes = list()
            for metric_name in avg_metrics:
                avg_metric_min = min(avg_metrics[metric_name])
                avg_metric_min_idx = avg_metrics[metric_name].index(
                                        avg_metric_min)
                votes.append(avg_metric_min_idx)
                
            winner_idx, _ = Counter(votes).most_common(1)[0]
            
        elif self._decision_method \
            in self._error_metrics:
            avg_metric_min = \
                min(avg_metrics[self._decision_method])
            winner_idx = \
                avg_metrics[self._decision_method].index(
                    avg_metric_min)
                
        else:
            raise ValueError('Unknown fusion method')

            
        self._best_model_hparams = comb_list[winner_idx]['params']
        
        print('Best hyperparameters set: {}'.format(
            self._best_model_hparams))
        
        winner_avg_metrics = {key: avg_metrics[key][winner_idx] 
                                  for key in self._error_metrics}
        print('Best hyperparameters metrics: {}'
                  .format(winner_avg_metrics))
        print('---------------------------------------------------')
        # -------------------------------------------------------------
    
    def get_best_hyperparameters(self):
        if not self._best_model_hparams:
            raise Exception('Hyperparameters have not been tuned')
        return self._best_model_hparams
