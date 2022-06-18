# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 18:41:41 2022

@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

import requests


class PushoverManager(object):
    """
    A class that represents a manager for Pushover notifications.
    Notifications are generated on experiment's end or error.
    
    Attributes
    ----------
    user: str
        Pushover user key
    token: str
        Pushover API token
    """
    
    def __init__(self,
                 user,
                 token):
        
        self._user = user
        self._token = token
        
    
    def _push_message(self,
                     message):

            payload = {
                'token': self._token,
                'user': self._user,
                'message': message
                }
            requests.post('https://api.pushover.net/1/messages.json',
                          data=payload)
    
    
    def notify_experiment_end(self, 
                              dataset,
                              model,
                              elapsed_time
                              ):
        
        message = 'Experiment finished | Dataset: {} | Model: {} | Elapsed time: {}s'.format(
                        dataset, model, str(elapsed_time))
        self._push_message(message)
    
    
    def notify_experiment_error(self, 
                              dataset,
                              model,
                              error
                              ):
        
        message = 'Experiment error | Dataset: {} | Model: {} | Error: {}'.format(
                        dataset, model, error)
        self._push_message(message)