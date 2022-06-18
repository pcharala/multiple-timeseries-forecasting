# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 20:28:44 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

import json
import os

class Config(object):    
    
    def __init__(self, conf_dir):
        (self._gconf, self._dconf, self._mconf) = \
            Config._read_configs(conf_dir)
    
    
    @staticmethod
    def _read_configs(conf_dir):
        with open(os.path.join(conf_dir, "general.conf"),
                               encoding = 'utf-8') as f:
            general = json.load(f)
            
        with open(os.path.join(conf_dir, "datasets.conf"),
                  encoding = 'utf-8') as f:
            datasets = json.load(f)
        
        with open(os.path.join(conf_dir, "models.conf"),
                       encoding = 'utf-8') as f:
            models = json.load(f)
            
        return (general, datasets, models)
    

    def get_general_config(self):
        return self._gconf


    def get_model_config(self):
        return self._mconf
    

    def get_dataset_config(self):
        return self._dconf
    
    
    def get_config(self,
                   dataset_name,
                   model_name,
                   metrics,
                   decision_method,
                   summary_stats):
        
        config = self._gconf
        config['dataset'] = self._dconf[dataset_name]
        config['dataset']['name'] = dataset_name
        config['model'] = self._mconf[model_name]
        config['model']['name'] = model_name
        config['metrics'] = metrics
        config['decision_method'] = decision_method
        config['summary_stats'] = summary_stats
        
        return config
