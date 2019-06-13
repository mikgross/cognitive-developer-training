# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 08:46:02 2018

@author: msahan
"""



# First Task -- Importing libraries 
import matplotlib.pyplot as plt # importing libraries 
import pandas as pd # importing libraries


# Second Task -- Importing Neural Network libraries 
from keras.models import Sequential # Importing Feed Forward Neural Network
from keras.layers import Dense, Activation # Importing layers features
from keras.layers import Dropout # Importing Dropout technique 
    

if __name__=='__main__':
    
    # Second Task -- Importing data
    # Read xlsx document
    xl = pd.read_excel('Scores.xlsx') 
    
    # Get data from the document
    data = xl.get_values()

    # Get columns for the data: number of document
    docs = data[:,0].reshape(-1,1)
    
    # Get columns for the data: take scores
    scores = data[:,1].reshape(-1,1)
    
    # Fouth Task -- Creating Feed Forward Neural Network
    
    # Creating NN
    ffnn = Sequential()
    
    #Fifth Task -- Adding Neural Network Layers 
    
    # Adding first hidden layer with 10 neurons and sigmoid activation function 
    ffnn.add(Dense(10))
    ffnn.add( Activation('sigmoid'))
            
    # Adding second hidden layer with 10 neurons and sigmoid activation function
    ffnn.add(Dense(10))
    ffnn.add(Activation('sigmoid'))
    
    # Add output layer with sigmoid activation function 
    ffnn.add(Dense(1))
    ffnn.add(Activation('sigmoid'))
    
    # Sixth Task -- Defining Loss functon 
    
    # Mean squared error loss function with adam optimizer
    ffnn.compile(loss='mean_squared_error', optimizer='adam')
    
    
    # Seventh Task -- NN training
    # NN training with 10000 iterations
    ffnn.fit(data, scores, epochs = 10000, batch_size = 32)
    
    
    # Eighth Task -- NN fitting
    
    # Predict
    pred = ffnn.predict(data)    
    
    # Ninth Task -- Results plotting
    
    
    
    #Creating NN
    ffnn2 = Sequential()

    # Tenth Task -- Adding New Neural Network Layers 
    # Adding first hidden layer with 10 neurons and sigmoid activation function 
    ffnn2.add(Dense(10))
    ffnn2.add(Activation('sigmoid'))

    #Adding second hidden layer with 10 neurons and sigmoid activation function
    ffnn2.add(Dense(10))
    ffnn2.add(Activation('sigmoid'))
    

    #Output layer with sigmoid activatiob function 
    ffnn2.add(Dense(1))
    ffnn2.add(Activation('sigmoid'))
    
    # Eleventh Task -- Defining Loss functon 
    # Mean squared error loss function with adam optimizer
    ffnn2.compile(loss='mean_squared_error', optimizer='adam')

    # Twelveth Task -- NN training 
    # NN training with 700 iterations
    ffnn2.fit(data, scores, epochs = 700, batch_size = 32)
    pred2 = ffnn2.predict(data)
    
    # Plotting of the Scores data
    plt.plot(scores, '.')

    #Plotting NN fit
    plt.plot(pred)

    # x-axis label
    plt.xlabel('number of document')

    #y-axis label
    plt.ylabel('scores')
    #Plotting NN fit
    plt.plot(pred2)
    
    ffnnM = ffnn.evaluate(data, scores, verbose = 1)
    ffnn2M = ffnn2.evaluate(data, scores, verbose = 1)
    