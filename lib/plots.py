import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def factor_plot(X, factors, prediction, color="Set3"):
    # First, plot the total for each factor. Then, plot the total for each
    # factor for the prediction variable (so in a conversion example, how
    # many people converted, revenue per country, etc.)
    
    # These refer to the rows and columns of the axis numpy array; not the 
    # data itself.
    
    row = 0
    column = 0
    sns.set(style="whitegrid")
    # TODO: Set the width based on the max number of unique 
    # values for the factors.
    
    plots = plt.subplots(len(factors), 2, figsize=(8,12))
    # It should 
    for factor in factors:
        sns.countplot(x=factor, palette="Set3", data=X, 
                      ax=plots[1][row][column])
        # Then print the total for each prediction
        sns.barplot(x=factor, y=prediction, data=X, ax=plots[1][row][column+1])
        row += 1
    plt.tight_layout() # Need this or else plots will crash into each other

def continuous_plots(data, continuous_factors):
    plots = plt.subplots(len(continuous_factors), 1, figsize=(8,12))
    row = 0
    for factor in continuous_factors:
        sns.distplot(data[factor], ax=plots[1][row])
        row += 1
    plt.tight_layout()

def summary_plots(data):
    # Scatter Plot of the factors color coded as based on the prediction 
    # variable's prevelence. 
    # Heatmap of the variable's against each other to show the correlations.
    # Geographical representation of the prediction variable (if applicable)
    # This will generate all summary plots needed for the data.
    # Histogram of the prediction variable with an overlayed density 
    # Stem and Leaf Plots of the variable across the factors
    # Line graph of the prediction variable over time (if applicable)
    # Quantiles of the prediction variable plotted
    # Box and Whisker plots of the data to show skewness or heavy tails
    # QQ Plot to also show skewness and heavy tails
    # 
    
    # First, do a pairplot of all of the variables
    sns.pairplot(data)
    
    # Then, you can do a heatmap to show the correlations a bit more in depth
    
    # 
    
    # Now we go a bit more in depth and show the distribution of each continuous
    # variable
    
    
    
    for column in data:
        """
        if isinstance(data[column][0], basestring):
            print "You didn't clean the data first...... Go clean that data" +\
            " first!"
            continue
            #sys.exit()
        # If it's categorical
        elif isinstance(data[column][0], int) and data[column].unique == 2:
            sns.stripplot(x=column, data=data)
        else:
            # it must be continuous
            sns.distplot(data[column])
        """
        continue