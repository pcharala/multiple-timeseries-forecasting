# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:04:18 2022

@author: Christos Tzagkarakis <tzagarak@ics.forth.gr>
@author: Pavlos Charalampidis <pcharala@ics.forth.gr>
"""

import pandas as pd
import numpy as np

from common.data_loader_Monash import convert_tsf_to_dataframe
from common.Utils import Utils



class DatasetLoader(object):
    """
    A class trat represents a dataset loader.
    
    Attributes
    ----------
    dataset_name: str
        the name of the dataset.
    dataset_path: str
        the dataset path.
    """
    
    
    def __init__(self,
                 dataset_name,
                 dataset_path):
        
        self._dataset_name = dataset_name
        self._dataset_path = dataset_path
        
    
    def dataset_load_and_preprocess(self):
        """
        Loads, preprocesses and returns dataset as ndarray.

        Raises
        ------
        ValueError
            If unknow dataset name.

        Returns
        -------
        ndarray
            Rows correspond to timeseries and columns to time steps.

        """
        
        if self._dataset_name == 'SanFranciscoTraffic':
            
            (df,
             frequency,
             forecast_horizon,
             contain_missing_values,
             contain_equal_length) = \
                convert_tsf_to_dataframe(self._dataset_path)
                
            original_ts = \
                np.zeros((df.shape[0], len(df.iloc[0,2].to_numpy())))
            for col in range(df.shape[0]):
                tmp_series = (df.iloc[col,2]).to_numpy().reshape(-1,1)
                original_ts[col,:] = np.transpose(tmp_series)
                
        elif self._dataset_name == 'GuangzhouTraffic':
            
            df = pd.read_csv(self._dataset_path, header=None)
            original_ts = df.to_numpy()
            
        elif self._dataset_name == 'ElectricityLoadDiagrams':
            
            df = pd.read_csv(self._dataset_path)
            
            # Drop the columns that do not contain any valuable information
            df.drop(df.columns[[0, 1, 322, 323, 324]],
                    axis = 1,
                    inplace = True)
            
            # Convert the dataframe into a 2D numerical array
            orig_ts = df.to_numpy()

            # Rows should correspond to the time-series and columns correspond to the time-stamps
            original_ts = np.transpose(orig_ts)

        elif self._dataset_name == 'EnergyConsumptionFraunhofer':
            
            df = pd.read_csv(self._dataset_path)

            df = df.iloc[: , 1:]  # Remove first dataframe's column since it contains the dates

            # Convert the dataframe into a 2D numerical array
            orig_ts = df.to_numpy()

            # Rows should correspond to the time-series and columns correspond to the time-stamps
            original_ts = np.transpose(orig_ts)
            
        elif self._dataset_name == 'LondonSmartMeters':
            
            df = pd.read_csv(self._dataset_path)

            # Select dates in the range below
            # 15024    2012-10-23 00:00:01
            # 25007    2013-05-18 23:30:01
            df = df.iloc[15024:25007, :]
            del df["Unnamed: 0"]
            del df["timestamps"]
            
            # Convert the dataframe into a 2D numerical array
            orig_ts = df.to_numpy()

            # Rows should correspond to the time-series and columns correspond to the time-stamps
            original_ts = np.transpose(orig_ts)

        else:
            
            raise ValueError('Unknown dataset: ' + self._dataset_name)

        Utils.print_dataset_info(df, self._dataset_name)
        Utils.print_ts_info(original_ts)

        return original_ts