# -*- coding: utf-8 -*-
"""
Created on Tue May 15 09:12:35 2018

@author: msahan
"""

# First Task -- Import libraries

import sklearn as sk # importing libraries 
import matplotlib.pyplot as plt # importing libraries
import pandas as pd # importing libraries

if __name__=='__main__':

    # Second Task -- Get data from the xlsx document
    
    # read xlsx document
    xl = pd.read_excel('Scores.xlsx') 
    
    # extraxt data
    data = xl.get_values()
    
    # get separate columns from data
    number_of_docs = data[:,0].reshape(-1,1)
    
    # get separate columns from data
    scores = data[:,1].reshape(-1,1)
        
    # Third Task -- Construct Linear Regression
    
    # linear regression object
    reg = sk.linear_model.LinearRegression()
    #fitting regression
    reg.fit(number_of_docs, scores) 
    #creating regression line
    predict = reg.predict(number_of_docs) 
    
    
    #Fourth Task   Plotting initial data 
    #plot of the scores data
    plt.plot(number_of_docs, scores,'.')
    #plot of the regression line
    plt.plot(number_of_docs, predict)
    #x-axis label
    plt.xlabel('Number of train documents') 
    #y-axis label
    plt.ylabel('Score of the method') 
    
    
    #Fifth Task   Determining the coefficiants 
    #cofficients of the Linear Regression
    reg.coef_ 
    
    
    #Sixth Task   Preparing data for polynomial linear regression 
    #Importing object for transformation
    transform = sk.preprocessing.PolynomialFeatures(3) 
    #Transformation of the data
    number_of_docs_transformed = transform.fit_transform(number_of_docs) 
    
    
    #Ninth Task   Importing Linear Regression
    #Importing Linear regression
    reg = sk.linear_model.LinearRegression()
    
    #Fitting regression
    reg.fit(number_of_docs_transformed, scores)
    
    #Creating regression line
    predict = reg.predict(number_of_docs_transformed) 
    
    #Tenth Task   Plotting results 
    #Plot of the non-linear Scores data
    plt.plot(number_of_docs, scores, '.') 
    #Plot of the regression curve
    plt.plot(number_of_docs, predict) 
    #x-axis label
    plt.xlabel('Number of train documents') 
    #y-axis label
    plt.ylabel('Score of the method') 
    
    
    #Eleventh Task   Determining the coefficiants 
    #cofficients of the Linear Regression
    reg.coef_ 
