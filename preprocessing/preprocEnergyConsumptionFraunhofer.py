# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 13:03:43 2022

Prediction of Energy Consumption for Variable Customer Portfolios Including Aleatoric Uncertainty Estimation
https://github.com/deepinsights-analytica/ieee-icpse2021-paper

Fraunhofer Fordatis database (https://fordatis.fraunhofer.de/handle/fordatis/215)

@author: Christos Tzagkarakis <tzagarak@ics.forth.gr>
"""

import pandas as pd
import holidays as hd


def preprocess(input_filepath,
               output_filepath):
    
    #%% Configuration
    categories = ['consumption', 'weather', 'profiles']
    files = input_filepath  
    
    # Function Definitions
    def fix_DST(data):
        data = data[~data.index.duplicated(keep='first')]  # Identify duplicated row indexes, and remove them (if so)
        data = data.resample('H').ffill()  # Resample data hourly, using forward filling
        return data
    
    def crop(data):
        hour_index = data.index.hour
        t0 = data[hour_index==0].head(1).index
        tn = data[hour_index==23].tail(1).index
        data.drop(data.loc[data.index < t0[0]].index, inplace=True)  # Remove entries corresponding to recorded data before '2019-01-01 00:00:00'
        data.drop(data.loc[data.index > tn[0]].index, inplace=True)  # Remove entries corresponding to recorded data after '2019-12-31 23:00:00'
        return data
    
    
    #%% Loading Consumption Data
    consumptions = pd.read_excel(files[categories.index('consumption')],
                                  parse_dates = [0],
                                  index_col = 0)
    consumptions.columns = pd.DataFrame(consumptions.columns, columns=['customer']).index
    consumptions.index.name = 'time'
    
    consumptions = fix_DST(consumptions)
    # The consumptions dataframe contains 499 columns. A column represents a single customer with a customer ID in the range
    #of 0...498. Each row in the dataframe contains the consumption values for a certain timestamp within the measurement
    #period of one year (Jan 01 - Dec 31, 2019). The time series were recorded with an hourly resolution. This means the dataframe
    #contains 8760 rows (365 day x 24 hours).
    
    consumptions = crop(consumptions)
    
    
    # Calculate day categories
    # Since the energy consumption behavior depends on weekly seasonalities like working day or public holiday, a
    #list of all public holidays is needed for feature extraction
    holidays = hd.ES(years=list(range(2010, 2021)), prov="MD")  # Loading Spanish holidays
    
    days = pd.DataFrame(pd.to_datetime(consumptions.index.date),
                        index = consumptions.index,
                        columns = ['date'])
    days['day_of_week'] = list(days.index.dayofweek)
    days['day_of_month'] = list(days.index.day)
    days['month'] = list(days.index.month)
    days['day_category'] = days['day_of_week'].replace({0:0, 1:1, 2:1, 3:1, 4:2, 5:3, 6:4})
    days.loc[days['date'].apply(lambda d: d in holidays), 'day_category'] = 4
    days = days.groupby(['date']).first()
    
    # The dataframe days contains additional information for characterizing a certain date (day). For each day
    #within the measurement period the day of the week, the day within the month, the month as well as the day
    #category was calculated. The used day categories are 'Monday' (0), 'Tuesday-Thursday' (1), 'Friday' (2),
    #'Saturday' (3), and 'Sunday or Holiday' (4).
    
    
    # The dataframe consumptions_daily_mean contains the daily mean values for the energy consumption of all
    #customers. This means the table has 499 columns (customers) and 365 rows (days within the measurement period).
    consumptions_daily_mean = pd.DataFrame(consumptions.groupby(consumptions.index.date).mean(),
                                            index = days.index)
    
    #%% Isolate only household energy consumptions
    customers = pd.read_excel(files[categories.index('profiles')])
    customers.columns = ['customer', 'profile']
    profiles = pd.DataFrame(customers['profile'].unique(), columns=['profile'])
    
    households = customers[customers['profile'].astype(str).str.contains('hogares')].index.values
    consumptions_households = consumptions.loc[:,households]
    
    consumptions_households_daily_mean = pd.DataFrame(consumptions_households.groupby(consumptions_households.index.date).mean(),
                                                      index = days.index)
    
    # the time spanning is: 1/1/2019 to 31/12/2019
    consumptions_households_daily_mean.to_csv(output_filepath)
