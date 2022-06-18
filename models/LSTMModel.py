# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 01:29:13 2022

@author: Christos Tzagkarakis <tzagarak@ics.forth.gr>
@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

from .BaseModel import BaseModel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras import backend
#from tensorflow.keras.models import load_model
#from tensorflow.keras.utils import plot_model


import importlib

class LSTMModel(BaseModel):
    """
    A class that represents an LSTM model.
    
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


    def _build_model(self, input_shape, output_shape):
        
        layers = [Input(shape=input_shape)]
        
        for idx, layer in enumerate(self._params['layers']):
            for i in range(layer['multiplicity']):
                if layer['type'] == 'lstm':
                    cur_layer = LSTM(units=layer['units'],
                                     **layer['keyargs'],
                                     return_sequences=True)
                elif layer['type'] == 'dense':
                    # check if last 2 layers include LSTM.
                    # if yes, set return_sequences=False
                    for l in layers[::-1][:2]:
                        if l.__class__.__name__ == 'LSTM':
                            l.return_sequences=False
                            break  
                    cur_layer = Dense(units=layer['units'],
                                      **layer['keyargs'])
                else:
                    raise ValueError('Unknown layer type')
                
                layers.append(cur_layer)
    
                try:
                    layers.append(Dropout(layer['dropout']))
                except KeyError:
                    pass
            
        for l in layers[::-1][:2]:
            if l.__class__.__name__ == 'Dense':
                break
            elif l.__class__.__name__ == 'LSTM':
                l.return_sequences=False
                break
            
        layers.append(Dense(units=output_shape))

        md = Sequential()
        for layer in layers:
            # print('-------------------------')
            # print(layer)
            # print('-------------------------')
            md.add(layer)
            
        md.compile(optimizer=self._params['optimizer'],
                   loss=self._params['loss'])

        return md


    def fit(self, train_data):
        
        (train_X, train_y) = train_data
        
        if self._trained_model:
            backend.clear_session()
            self._trained_model = None
        model = self._build_model(train_X.shape[1:],
                                  train_X.shape[2])
        # model.summary()
        # plot_model(model,
        #            to_file='lstm.png',
        #            show_shapes=True,
        #            show_layer_names=True)
        
        cb_list = list()
        try:
            for cb in self._params['callbacks']:
                cb_module = importlib.import_module(
                    'tensorflow.keras.callbacks')
                cb_class_ = getattr(cb_module, cb['name'])            
                cb_copy = cb.copy()
                cb_copy.pop('name')
                cb_instance = cb_class_(**cb_copy)
                cb_list.append(cb_instance)
        except KeyError:
            pass

        model.fit(train_X,
                  train_y,
                  epochs=self._params['epochs'],
                  batch_size=self._params['batch_size'],
                  callbacks=cb_list,
                  verbose=0)
   
        self._trained_model = model


    def predict(self, test_data):
        
        (test_X, test_y) = test_data
        
        if self._trained_model:
            return self._trained_model.predict(test_X, verbose=0)
        raise ValueError('Model is not trained')

