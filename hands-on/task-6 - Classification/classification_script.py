# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 08:46:02 2018

@author: msahan
"""



#First Task   Importing libraries
import data_processing_functions as mfc #importing library
import numpy as np #importing numpy lobrary
#Fourth Task   Importing classification methods
from sklearn import svm #Support Vector Machines method
from sklearn.naive_bayes import GaussianNB #Naive Bayes method
from sklearn.linear_model import LogisticRegression #Logistic Regression
    

if __name__=='__main__':
    #Second Task   Creating bag-of-words
    directory_path = "..\\task-3 - KMeans\\exercise\\full_dataset\\"
    #Creating Bag-of-words object
    bag_of_words = mfc.BagOfWords(directory_path,True,True,True)
    #Creating Bag-of-words
    bag_of_words_matrix = bag_of_words.create_bag_of_words() 
    
    
    #Third Task   Creating Features vector with class
    #Creating Features vector
    features_vector = bag_of_words.create_features_vector() 
    
    
    #Fifth task   Creating classification objects
    #Creating SVM
    svm_object = svm.LinearSVC() 
    #Creating Naive Bayes
    nb_object = GaussianNB() 
    #Creating Logistic Regression
    lreg_object = LogisticRegression() 
    
    
    #Sixth Task   Dataset shuffle
    shuffled_bag_of_words, shuffled_features_vector = bag_of_words.shuffle_array(bag_of_words_matrix, features_vector)
    
    #Amount of Test docs   
    amount = 400 
    
    #Seventh Task   Classifiers training 
    #SVM training 
    svm_object.fit(shuffled_bag_of_words[0:amount,:], shuffled_features_vector[0:amount]) 
    #NB training 
    nb_object.fit(shuffled_bag_of_words[0:amount,:], shuffled_features_vector[0:amount]) 
    #Logistic Regression training
    lreg_object.fit(shuffled_bag_of_words[0:amount,:], shuffled_features_vector[0:amount]) 
    
    
    #Eights Task   Classifier testing  
    #SVM testing
    res1 = svm_object.predict(shuffled_bag_of_words[400:,:]) 
    #NB testing  
    res2 = nb_object.predict(shuffled_bag_of_words[400:,:])
    #Logistic Regression 
    res3 = lreg_object.predict(shuffled_bag_of_words[400:,:]) 
    
    #SVM scores 
    scores_SVM = svm_object.score(shuffled_bag_of_words[400:,:], shuffled_features_vector[400:]) 
    #NB scores
    scores_NB = nb_object.score(shuffled_bag_of_words[400:,:], shuffled_features_vector[400:])
    #NN scores
    scores_Lreg = lreg_object.score(shuffled_bag_of_words[400:,:], shuffled_features_vector[400:]) 
    
    
    #Ninth Task   Different kernerls SVM
    svm_non_lin_kernel_object = svm.SVC(decision_function_shape='ovo', degree=3, kernel='rbf')
    svm_non_lin_kernel_object.fit(shuffled_bag_of_words[0:amount,:], shuffled_features_vector[0:amount])
    #SVM testing
    res11 = svm_non_lin_kernel_object.predict(shuffled_bag_of_words[400:,:]) 
    #SVM scores 
    scores_SVM_non_lin = svm_non_lin_kernel_object.score(shuffled_bag_of_words[400:,:], shuffled_features_vector[400:]) 
    
    
    #Tenth Task   Different NB distributions
    from sklearn.naive_bayes import BernoulliNB
    bernoulli_Nb = BernoulliNB()
    #NB training 
    bernoulli_Nb.fit(shuffled_bag_of_words[0:amount,:], shuffled_features_vector[0:amount])
    #NB testing  
    res22=bernoulli_Nb.predict(shuffled_bag_of_words[400:,:]) 
    #NB scores
    scores_NB_bernoulli = bernoulli_Nb.score(shuffled_bag_of_words[400:,:], shuffled_features_vector[400:]) 
    
    from sklearn.naive_bayes import MultinomialNB
    multinomial_Nb = MultinomialNB()
    #NB training
    multinomial_Nb.fit(shuffled_bag_of_words[0:amount,:], shuffled_features_vector[0:amount])
    #NB testing
    res22 = multinomial_Nb.predict(shuffled_bag_of_words[400:,:])
    #NB scores
    Scores_NB_multinomial=multinomial_Nb.score(shuffled_bag_of_words[400:,:], shuffled_features_vector[400:]) 
    
