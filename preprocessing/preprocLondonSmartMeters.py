# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 18:18:38 2022

Monash London smart meter half-hourly dataset pre-processing

# https://zenodo.org/record/4656091#.YfKSMupBxaR

@author: Christos Tzagkarakis <tzagarak@ics.forth.gr>
"""

import pandas as pd
import numpy as np
from common.data_loader_Monash import convert_tsf_to_dataframe


def preprocess(input_filepath,
               output_filepath):
    
    (df,
     frequency,
     forecast_horizon,
     contain_missing_values,
     contain_equal_length) = convert_tsf_to_dataframe(input_filepath[0])
    
    
    ts_length = []
    for col in range(df.shape[0]):
        ts_length.append(len((df.iloc[col,2]).to_numpy().reshape(-1,1)))
    
    ts_length = np.array(ts_length)
    
    # Time series lengths in range 28272--28944
    ts_specific_length = [28272, 28320, 28368, 28416, 28464, 28512, 28560, 28608, 28656, 28704, 28752, 28800, 28848, 28896, 28944]
    
    df_list = []
    for l in range(len(ts_specific_length)):
        # Find original dataframe 'df' rows indexes whose time-series has exactly this length
        tmp_ts = np.where(ts_length == ts_specific_length[l])
        
        # Then isolate this specific sub-dataframe
        tmp_df = df.iloc[tmp_ts[0], :]
        
        df_list.append(tmp_df)
    
    # Concatenate the sub-dataframes
    df_conc = pd.concat(df_list)
    df_conc.reset_index(drop=True, inplace=True)  # Reset index
    
    #--------
    idx = pd.date_range('2011-12-15 00:00:01', periods=40000, freq="0.5H")
    ts_conc = pd.Series(range(len(idx)), index=idx)
    ts_conc = pd.DataFrame(ts_conc)
    ts_conc.reset_index(level=0, inplace=True)
    ts_conc = ts_conc.rename(columns={"index": "timestamps"})
    ts_conc.drop(ts_conc.columns[[1]], axis = 1, inplace = True)
    
    df_conc_TS = pd.DataFrame(np.zeros((40000, df_conc.shape[0])))
    df_conc_TS.insert(0, "timestamps", ts_conc['timestamps'], True)
    
    # Here we place each  time series vertically in the dataframe 'df_conc_TS', in a chronological
    #order based on each time series starting timestamp. It's shape is 40,000 x 505, where the first
    #column contains the timestamps (half-hourly) and the rest 504 columns contains the time series.
    for ts_iter in range(df_conc.shape[0]):
        ts_current = (df_conc.iloc[ts_iter, 2]).to_numpy().reshape(-1,1)
        start_date = df_conc.iloc[ts_iter, 1]
        start_index = (df_conc_TS[(df_conc_TS['timestamps']) == start_date].index)[0]
        df_conc_TS.loc[start_index:start_index+len(ts_current)-1, ts_iter] = ts_current
    
    
    # the time spanning is: 2011-12-15 00:00:01 to 2014-03-27 07:30:01
    df_conc_TS.to_csv(output_filepath)

