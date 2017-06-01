import pandas as pd
import numpy as np
import sys
import seaborn as sns

def dummy_variable_check(data, column, dummies, ranges=None, variable=None):
    flag = True
    if variable is 'Basic': # This is a basic 0, 1 dummy variable
        for dummy in dummies:
            # We just make sure that for each factor, there are enough rows for
            # that particular dummy variable.
            flag = (data.loc[data[column] == dummy].shape[0] == \
            data.loc[data[dummy] == True].shape[0])
            if flag is False:
                print ("Data Lists are not equal. Check Features.")
                return None
        print ("All Variables are OK for", column, "column")
    else: # This is for checking numeric ranges
        first_dummy = dummies[0]
        last_dummy = len(dummies)-1
        first_range = ranges[0]
        last_range = ranges[len(ranges)-1]
        count = 0
        flag = (data.loc[data[column] <= first_range].shape[0] == \
            data.loc[data[first_dummy] == True].shape[0])
        for i in range(1, len(dummies)-1):
            flag = data.loc[(data[column] > ranges[count]) &
                            (data[column] < ranges[count+1])].shape[0] == \
                            data.loc[data[dummies[i]] == True].shape[0]
            count += 1
        flag = (data.loc[data[column] > last_range].shape[0] == \
            data.loc[data[dummies[last_dummy]] == True].shape[0])
        if flag == False:
            print ("Data Lists are not equal. Check Features.")
            return None
        else:
            print ("All Variables are OK for", column, "column")

def dummy_variable(data, column):
    # Get the unique factors in that column
    dummies = data[column].unique()
    for variable in dummies:
        # turn that factor into a boolean.
        data[variable] = data[column] == variable
    # Check to see if the dummy variables were created properly.
    dummy_variable_check(data, column, dummies, ranges=None, 
                         variable='Basic')
    
    # Drop the old column with the factor
    data = data.drop(column, axis=1, inplace=False)
    return data*1

def numeric_dummy_variable(data, column, column_names, ranges):
    minimum = data[column].min()
    maximum = data[column].max()
    break_point = maximum / len(column_names)
    
    data[column_names[0]] = data[column] <= minimum
    if ranges:
        data[column_names[0]] = data[column] <= ranges[0]
        for i in range(len(ranges)-1):
            data.loc[(data[column] > ranges[i]) & (data[column] < ranges[i+1]),
                 column_names[i+1]] = True
            data[column_names[i+1]].fillna(value=False, inplace=True)
        data[column_names[len(column_names)-1]] = \
        data[column] > ranges[len(column_names)-2]
        dummy_variable_check(data, column, column_names, ranges=ranges,
                             variable=False)
        data = data.drop(column, axis=1, inplace=False)
        return data*1
    else:       
        for i in range(1, column_names-1):
            data.loc[(data[column] > break_point) \
                     and (data[column] < break_point*2),
                     column_names[i]] = True
            data[column_names[i]].fillna(value=False, inplace=True)
            break_point *= 2
        data = data.drop(column, axis=1, inplace=False)

def stratified_sampling(data, prediction, samples, class_imbal_tol):
    total = data.shape[0]
    grouped_data = data.groupby(by=[prediction])
    # Calculate the disparity
    newFrame = pd.DataFrame()
    count = 0
    for index, group in grouped_data:
        if (group.shape[0] / float(total)) <= class_imbal_tol:
            print ("Class Imbalance Detected!")
            for index, group in grouped_data:
                sample = group.sample(n=int(total*samples[count]),
                                      replace=True)
                newFrame = pd.concat([newFrame, sample])
                count += 1
            # Check the new dataFrame to ensure they are balanced
            #ax = sns.countplot(x=prediction, data=data)
            print ("New Class Distributions")
            print ("1: ", (newFrame.loc[newFrame[prediction] == 1].shape[0]) / float(newFrame.shape[0]))
            print ("0: ",(newFrame.loc[newFrame[prediction] == 0].shape[0]) / float(newFrame.shape[0]))
            return newFrame