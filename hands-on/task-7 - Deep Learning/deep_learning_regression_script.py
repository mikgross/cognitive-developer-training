# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 08:46:02 2018

@author: msahan
"""



#First Task   Importing libraries 
import matplotlib.pyplot as plt #importing libraries 
import pandas as pd #importing libraries


#Third Task    #Importing Neural Network libraries 
from keras.models import Sequential #Importing Feed Forward Neural Network
from keras.layers import Dense #Importing layers features
from keras.layers import Dropout #Importing Dropout technique 
    

if __name__=='__main__':
    #Second Task   Importing data
    #Read xlsx document
    xl = pd.read_excel('Scores.xlsx') 
    #Get data from the document
    data=xl.get_values() 
    #Get columns for the data  
    number_of_docs=data[:,0].reshape(-1,1) 
    #Get columns for the data
    scores=data[:,1].reshape(-1,1) 
    
    
    #Fouth Task   #Creating Feed Forward Neural Network
    #Creating NN
    model = Sequential() 
    
    
    #Fifth Task   #Adding Neural Network Layers 
    #Adding first hidden layer with 10 neurons and sigmoid activation function 
    model.add(Dense(20, activation='sigmoid', input_dim=1))
    #Adding second hidden layer with 10 neurons and sigmoid activation function
    model.add(Dense(20, activation='sigmoid'))
    #Output layer with sigmoid activatiob function 
    model.add(Dense(1, activation='sigmoid'))
    
    
    
    #Sixth Task   #Defining Loss functon 
    #Mean squared error loss function with adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    
    #Seventh Task   #NN training
    #NN training with 10000 iterations
    model.fit(number_of_docs,scores, epochs=5000, batch_size=None)
    
    
    #Eighth Task   #NN fitting
    #Predict
    predicted=model.predict(number_of_docs) 
    
    
    #Ninth Task   #Results plotting
    #Plotting of the Scores data
    plt.plot(number_of_docs,scores,'.')
    #Plotting NN fit
    plt.plot(number_of_docs,predicted)
    #x-axis label
    plt.xlabel('Number of train documents')
    #y-axis label
    plt.ylabel('Score of the method') 
    
    
    #Creating NN
    model = Sequential()
    
    
    #Tenth Task   #Adding New Neural Network Layers 
    #Adding first hidden layer with 10 neurons and sigmoid activation function 
    model.add(Dense(20, activation='sigmoid', input_dim=1))
    model.add(Dropout(0.2))
    #Adding second hidden layer with 10 neurons and sigmoid activation function
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dropout(0.2))
    #Output layer with sigmoid activatiob function 
    model.add(Dense(1, activation='sigmoid'))
    
    
    #Eleventh Task   #Defining Loss functon 
    #Mean squared error loss function with adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam') 
    
    
    #Twelveth Task   #NN training 
    #NN training with 700 iterations
    model.fit(number_of_docs,scores, epochs=5000, batch_size=None)
    
    
    #Eighth Task   #NN fitting
    predicted=model.predict(number_of_docs)
    
    
    #Ninth Task   #Results plotting
    #Plotting of the Scores data
    plt.plot(number_of_docs,scores,'.')
    #Plotting NN fit
    plt.plot(number_of_docs,predicted) 
    #x-axis label
    plt.xlabel('Number of train documents')
    #y-axis label
    plt.ylabel('Score of the method')