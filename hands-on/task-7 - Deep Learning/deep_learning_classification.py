# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:06:44 2018

@author: msahan
"""

#First Task   Importing libraries
import data_processing_functions as mfc #importing library
import numpy as np #importing numpy lobrary
#Importing Feed Forward Neural Network    
from keras.models import Sequential 
#Importing layers features
from keras.layers import Dense 
    


if __name__=='__main__':
    
    #Second Task   Loading Bag-of-words for classification 
    directory_path = '..\\0. Data\\full_dataset\\' 
    bag_of_words = mfc.BagOfWords(directory_path, True, True, True)
    bag_of_words_matrix = bag_of_words.create_bag_of_words()
    
    #Third Task   Creating Features vector with class 
    features_vector = bag_of_words.create_features_vector() #Creating Features vector
    
    
    shuffled_bag_of_words, shuffled_features_vector = bag_of_words.shuffle_array(bag_of_words_matrix, features_vector)
    
    
    shuffled_fetures_vector_for_keras = np.zeros((len(features_vector),4))
    
    
    for i in range(len(shuffled_features_vector)):
        if shuffled_features_vector[i]==1:
            shuffled_fetures_vector_for_keras[i,0]=1
        if shuffled_features_vector[i]==2:
            shuffled_fetures_vector_for_keras[i,1]=1
        if shuffled_features_vector[i]==3:
            shuffled_fetures_vector_for_keras[i,2]=1
        if shuffled_features_vector[i]==4:
            shuffled_fetures_vector_for_keras[i,3]=1
        
    
    #Fouth Task   #Creating Feed Forward Neural Network
    #Creating NN
    model = Sequential() 
    
    
    #Fifth Task   #Adding Neural Network Layers 
    #Adding first hidden layer with 50 neurons and sigmoid activation function 
    model.add(Dense(50, activation='sigmoid', input_dim=len(shuffled_bag_of_words[0])))
    #Adding second hidden layer with 50 neurons and sigmoid activation function
    model.add(Dense(50, activation='sigmoid'))
    #Output layer with softmax activatiob function 
    model.add(Dense(4, activation='softmax')) 
    
    
    #Sixth Task   #Defining Loss functon 
    #Mean squared error loss function with adam optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 
    
    
    #Seventh Task   #NN training
    #NN training with 100 iterations
    model.fit(shuffled_bag_of_words[:400,:],shuffled_fetures_vector_for_keras[:400,:], epochs=100, batch_size=None) 
    
    
    #Eighth Task   #NN fitting
    predicted = (model.predict(shuffled_bag_of_words[400:,:]))
    features_vector_for_comparison = shuffled_fetures_vector_for_keras[400:,:]