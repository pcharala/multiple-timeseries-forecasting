# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 19:03:16 2022

Electricity dataset pre-processing

@author: Christos Tzagkarakis <tzagarak@ics.forth.gr>
"""

import numpy as np
import pandas as pd
from datetime import datetime as dt
import copy
# import matplotlib.pyplot as plt


def preprocess(input_filepath,
               output_filepath):
    
    df = pd.read_csv(input_filepath[0], sep=";")

    # Check each column type
    # for col in df.columns:
    #     print(col, type(df[col].iloc[0]))
    
    # We can see that there are three types: int, str and numpy.float64
    
    # We have to replace the decimal comma with the dot, of all decimals in the dataframe
    df = df.replace(r",", ".",regex = True)
    
    # Rename column 'Unnamed: 0' to 'time'
    df.rename(columns = {"Unnamed: 0": "time"}, inplace = True)
    
    # Since everything except 'time' is a decimal type, separate 'time' once and convert the rest of the data into float type
    data = df.drop("time", axis = 1)
    data = data.astype("float")
    
    # Values are in kW of each 15 min. To convert values in kWh values must be divided by 4.
    data = data/4
    data["time"] = df["time"]
    
    # Functions used to help us to extract the month, day, day of the week, and hour from the 'time' column
    def get_year_from_time(x):
        time_date = dt.strptime(x, "%Y-%m-%d %H:%M:%S")
        return time_date.year
    
    def get_month_from_time(x):
        time_date = dt.strptime(x, "%Y-%m-%d %H:%M:%S")
        return time_date.month
    
    def get_day_from_time(x):
        time_date = dt.strptime(x, "%Y-%m-%d %H:%M:%S")
        return time_date.day
    
    def get_weekday_from_time(x):
        time_date = dt.strptime(x, "%Y-%m-%d %H:%M:%S")
        return time_date.weekday()
    
    def get_hour_from_time(x):
        time_date = dt.strptime(x, "%Y-%m-%d %H:%M:%S")
        return time_date.hour
    
    # Apply the functions created above all at once with 'apply'
    data["year"] = data["time"].apply(get_year_from_time)
    data["month"] = data["time"].apply(get_month_from_time)
    data["day"] = data["time"].apply(get_day_from_time)
    data["weekday"] = data["time"].apply(get_weekday_from_time)
    data["hour"] = data["time"].apply(get_hour_from_time)
    
    # print('starting recording date:',data['time'].min())
    # print('ending recording date:',data['time'].max())
    
    # Check potential missing values
    # print('Data contain missing values?--> ', data.isnull().values.any())
    # We can see that no missing values are included in the dataframe
    
    
    # List of customers column names
    col_customer = list(data.columns)
    col_customer = col_customer[:-6]
    
    # Here, we identify the index of the first non-zero value per customer time-series
    starting_time_by_customer = []
    for col in col_customer:
        for i in range(len(data)):
            if data[col][i] != 0:
                starting_time_by_customer.append(i)
                # Append only the lines that are not 0 for the first time by breaking the loop
                # Since it starts from 'MT_001', it will be accumulated for each client number
                break
    
    # Convert the list to dataframe
    starting_time_by_customer = pd.DataFrame(starting_time_by_customer)
    
    # Change the column name to 'index'
    starting_time_by_customer.rename(columns={0:"index"}, inplace=True)
    
    # Copy data to data_for_starting_time
    data_for_starting_time = copy.copy(data)
    
    # Add 'index' as the first column
    data_for_starting_time = data_for_starting_time.reset_index()
    
    # Merge with data_for_starting_time with 'index' as key
    starting_time_by_customer = pd.merge(starting_time_by_customer,
                                         data_for_starting_time[["index","time"]],
                                         on = "index",
                                         how = "left")
    # # Plot histogram
    # plt.figure(figsize=(10,5), dpi=80)
    # plt.hist(starting_time_by_customer["time"], bins=50)
    # plt.xticks(rotation=80)
    # plt.show()
    
    
    # It is obvious from the histogram that many clients have started recording from January 1, 2012. Thus, we
    #will use the data from January 1, 2012 uniformly. Clients whose recording has started before January 1, 2012
    #will be excluded from the analysis. In other words, target clients have starting_time--> 2012/1/1 or earlier.
    customer_list_for_analysis = []
    for col in col_customer:
        # Column ID: it is the number the client name ('MT_' part)
        col_id = int(col[3:])
        
        # Substitute the recording start time from 'MT_001' into 'starting_time', and the number+1 in the leftmost
        #column becomes the client name (index+1)
        # Time needs to be changed to list type
        starting_time = dt.strptime(list(starting_time_by_customer[starting_time_by_customer.index+1==col_id]["time"])[0],
                                    "%Y-%m-%d %H:%M:%S")
        if starting_time < dt(2012,1,2):
            # if col != "MT_223" and col != "MT_347":
            customer_list_for_analysis.append(col)
    
    customer_list_for_analysis += ["time",
                                   "month",
                                   "day",
                                   "weekday",
                                   "hour"]
    
    selected_customer_data = copy.copy(data[customer_list_for_analysis])
    
    # Delete the last row, since it corresponds to the date 1/1/2015
    selected_customer_data = selected_customer_data.drop(selected_customer_data.shape[0]-1,
                                                         axis=0)
    
    # Since the target period is from 2012/1/1 and onwards, delete the first 24*4*365=35040 lines, too
    selected_customer_data = selected_customer_data.drop(np.arange(0,35040),
                                                         axis=0)
    
    # Compute the daily energy consumption of customers
    time_list = selected_customer_data["time"]
    
    # Extract only the date (split the blank of time)
    f = lambda x:x.split(" ")[0]
    tmp = time_list.transform(f)
    selected_customer_data["day"] = tmp
    
    # Compute the total amount of energy consumption per day
    groupby_day_selected_customer_data = selected_customer_data.groupby(["day"], as_index=False).sum()
    
    # plt.figure(figsize=(10,5), dpi=80)
    # plt.hist(groupby_day_selected_customer_data.mean(), bins=100)
    # plt.show()
    
    # the time spanning is: 1/1/2011 to 31/12-2014
    groupby_day_selected_customer_data.to_csv(output_filepath)
