# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:18:38 2022

Guangzhou traffic speed dataset pre-processing

@author: Christos Tzagkarakis <tzagarak@ics.forth.gr>
"""

import shutil


def preprocess(input_filepath,
               output_filepath):
    
    shutil.copyfile(input_filepath[0],
                    output_filepath)
