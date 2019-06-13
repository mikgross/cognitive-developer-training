# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:06:44 2018

@author: msahan
"""

# First Task -- Importing libraries
import data_processing_functions as mfc # importing library
import numpy as np # importing numpy lobrary
from numpy import random

# Importing Feed Forward Neural Network    
from keras.models import Sequential 

# Importing layers features
from keras.layers import Dense, Activation
from keras.utils import Sequence
import matplotlib.pyplot as plt
    
def generator(features, labels, batch_size):
       # Create empty arrays to contain batch of features and labels#
       batch_features = np.zeros((batch_size, 10000))
       batch_labels = np.zeros((batch_size,4))
       while True:
           for i in range(batch_size):
               # choose random index in features
               index = random.choice(len(features),1)
               batch_features[i] = features[index]
               batch_labels[i] = labels[index]
           yield batch_features, batch_labels

if __name__=='__main__':
    
    # Second Task -- Loading Bag-of-words for classification 
    directory = '.\\..\\task-3 - KMeans\\exercise\\full_dataset\\'
    bow = mfc.BagOfWords(directory, True, True, True)
    bowm = bow.create_bag_of_words()
    
    #Third Task   Creating Features vector with class 
    fv = bow.create_features_vector()
    
    shubowm, shufv = bow.shuffle_array(bowm, fv)
    
    # Why should we do this? shuffled_fetures_vector_for_keras = np.zeros((len(features_vector),4))
    kshufv = np.zeros((len(fv),4))
    
    for i in range(len(shufv)):
        if shufv[i]==1:
            kshufv[i,0]=1
        if shufv[i]==2:
            kshufv[i,1]=1
        if shufv[i]==3:
            kshufv[i,2]=1
        if shufv[i]==4:
            kshufv[i,3]=1
        
    # Fouth Task -- Creating Feed Forward Neural Network
    
    '''
    
    First model
    
    '''
    # Creating NN
    model = Sequential()
    
    # Fifth Task -- Adding Neural Network Layers
    # Adding first hidden layer with 50 neurons and sigmoid activation function 
    model.add(Dense(50))
    model.add(Activation('sigmoid'))

    # Adding second hidden layer with 50 neurons and sigmoid activation function
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    
    # Output layer with softmax activatiob function 
    model.add(Dense(4))
    model.add(Activation('softmax'))
    
    # Sixth Task -- Defining Loss functon
    # Mean squared error loss function with adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # Seventh Task -- NN training
    am = 400
    # NN training with 100 iterations
    model.fit(shubowm[:am,:], kshufv[:am, :], epochs = 400, batch_size = None)

    pred = model.predict(bowm[am:,:])

    score = model.evaluate(shubowm, kshufv)
    


    '''
    
    Second model
    
    '''
    # Creating NN
    model2 = Sequential()
    
    # Fifth Task -- Adding Neural Network Layers
    # Adding first hidden layer with 50 neurons and sigmoid activation function 
    model2.add(Dense(30))
    model2.add(Activation('relu'))

    # Adding second hidden layer with 50 neurons and sigmoid activation function
    model2.add(Dense(50))
    model2.add(Activation('relu'))
    
    # Adding second hidden layer with 50 neurons and sigmoid activation function
    model2.add(Dense(50))
    model2.add(Activation('relu'))
    
    # Adding second hidden layer with 50 neurons and sigmoid activation function
    model2.add(Dense(30))
    model2.add(Activation('relu'))
    
    # Output layer with softmax activatiob function 
    model2.add(Dense(4))
    model2.add(Activation('softmax'))
    
    # Sixth Task -- Defining Loss functon
    # Mean squared error loss function with adam optimizer
    model2.compile(loss='mean_squared_error', optimizer='adam')
    
    # Seventh Task -- NN training
    am2 = 2800
    ep2 = 1000
    BS = 300

    # NN training with 100 iterations
    model2.fit(shubowm[:am2,:], kshufv[:am2, :], epochs = ep2, batch_size = BS, verbose = 1)
    
    H = model2.fit_generator(
            generator(shubowm[:am2,:], kshufv[:am2, :], BS), 
            epochs = ep2, 
            steps_per_epoch = len(shubowm[:am2,:]) // BS, 
            validation_data=(shubowm[:am2,:], kshufv[:am2, :]), verbose = 1)
    
    
    pred2 = model2.predict(bowm[am2:,:])

    score2 = model2.evaluate(shubowm, kshufv)
    '''
    plt.plot(pred2, '.')
    
    plt.xlabel('Doc number')
    plt.ylabel('Doc number')    
    '''
    # graph
    plt.plot(np.arange(0, ep2), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, ep2), H.history["val_loss"], label="val_loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="down")
