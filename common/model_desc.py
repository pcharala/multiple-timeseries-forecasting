# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 12:55:42 2022

@author: pcharala
"""

def generate_model_description(model, hp):        
    if model == 'LSTMModel':
        model_desc = \
            {
                "layers": [
                    {
                        "type": "lstm",
                        "multiplicity": hp['lstm_mulitplicity'],
                        "units": hp['lstm_units'],
                        "keyargs": {
                            "activation": hp['lstm_activation'],
                        }
                    },
                    {
                        "type": "dense",
                        "multiplicity": hp['dense_mulitplicity'],
                        "units": hp['lstm_units'],
                        "keyargs": {
                            "activation": hp['dense_activation'],
                        }
                    }
                ],
                "optimizer": hp['optimizer'],
                "loss": hp['loss'],
                "epochs": hp['epochs'],
                "batch_size": hp['batch_size']
            }
    elif model == 'CNNModel':
        model_desc = \
            {
                "layers": [
                        {
                        "type": "conv1d",
                        "filters": hp['conv1d_filters'],
                        "kernel_size": hp['conv1d_kernel_size'],
                        "keyargs": {
                            'activation': hp['conv1d_activation']
                            }
                        },
                        {
                        "type": "maxpool1d",
                        "keyargs": {
                            'pool_size': hp['maxpool_size']
                            }
                            
                        }
                    ],
                "optimizer": hp['optimizer'],
                "loss": hp['loss'],
                "epochs": hp['epochs'],
                "batch_size": hp['batch_size']
                }
    elif model == 'BiLSTMModel':
        model_desc = \
            {
                "layers": [
                    {
                        "type": "lstm",
                        "units": hp['lstm_units'],
                        "keyargs": {
                            "activation": hp['lstm_activation'],
                        }
                    }
                ],
                "optimizer": hp['optimizer'],
                "loss": hp['loss'],
                "epochs": hp['epochs'],
                "batch_size": hp['batch_size']
            }
    elif model == 'DeepESNModel':
        model_desc = {
            'Nr': hp['Nr'], # recurrent units per layer
            'Nl': hp['Nl'], # recurrent layers
            'reg': hp['reg'], # readout reg.
            'transient': hp['transient'],
            'rhos': hp['rhos'], # spectral radius
            'lis': hp['lis'], # leaking rate
            'iss': hp['iss'], # input scaling
            'IPconf':{
                'DeepIP': 1,
                'threshold': 0.1,
                'eta': 1e-5,
                'mu': 0,
                'sigma': hp['ip_sigma'],
                'Nepochs': 10,
               },
            'reservoirConf': {
                'connectivity': 1
                },
            'readout': {
                'trainMethod': 'NormalEquations'
                }
        }
    elif model in ['BHTArimaModel',
                   'RFModel',
                   'SVRModel']:
        model_desc = hp
    
    return model_desc