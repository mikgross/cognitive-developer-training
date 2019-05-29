# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 08:46:02 2018

@author: msahan
"""

# First Task -- Importing libraries
import data_processing_functions as mfc #importing library
# Fourth Task -- Importing classification methods
from sklearn import svm #Support Vector Machines method
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB #Naive Bayes method
from sklearn.linear_model import LogisticRegression #Logistic Regression
    

if __name__=='__main__':

    # Second Task -- Creating bag-of-words
    directory_path = "..\\task-3 - KMeans\\exercise\\full_dataset\\"
    # Creating Bag-of-words object
    bag_of_words = mfc.BagOfWords(directory_path,True,True,True)
    # Creating Bag-of-words
    bag_of_words_matrix = bag_of_words.create_bag_of_words() 
    
    # Third Task -- Creating Features vector with class        
    # Creating Features vector
    feature_vetcor = bag_of_words.create_features_vector()
    
    # Fifth task -- Creating classification objects
    # Creating SVM: linear SVC used here 
    linearSVC = svm.LinearSVC()
    # Creating Naive Bayes
    naiveBayes = GaussianNB()    
    # Creating Logistic Regression
    logReg = LogisticRegression()
    
    # Sixth Task -- Dataset shuffle
    
    # we want to shuffle the bag of words and the features
    shuffBowMat, shuffFv = bag_of_words.shuffle_array(bag_of_words_matrix, feature_vetcor)
    # Amount of Test docs
    amount = 400
    
    # Seventh Task -- Classifiers training 
    
    # SVM training 
    linearSVC.fit(shuffBowMat[:amount], shuffFv[:amount])
    # NB training
    naiveBayes.fit(shuffBowMat[:amount], shuffFv[:amount])
    #Logistic Regression training
    logReg.fit(shuffBowMat[:amount], shuffFv[:amount])
    
    # Eights Task -- Classifier testing  
    # SVM testing
    predSVC = linearSVC.predict(shuffBowMat[amount:])
    
    # NB testing
    predNB = naiveBayes.predict(shuffBowMat[amount:])
    
    # Logistic Regression 
    predLogReg = logReg.predict(shuffBowMat[amount:])
    
    # SVM scores
    scoreSVM = linearSVC.score(shuffBowMat[amount:, :], shuffFv[amount:])
    
    # NB scores
    scoreNB = naiveBayes.score(shuffBowMat[amount:, :], shuffFv[amount:])
    
    # NN scores
    scoreLogReg = logReg.score(shuffBowMat[amount:, :], shuffFv[amount:])
    
    
    # Ninth Task -- Different kernerls SVM
    svmSVC = svm.SVC()
    svmSVC.fit(shuffBowMat[:amount], shuffFv[:amount])
    
    #SVM testing
    predSvmSVC = svmSVC.predict(shuffBowMat[:amount])
    
    #SVM scores 
    scoresvmSVC = svmSVC.score(shuffBowMat[amount:], shuffFv[amount:])
    
    # Tenth Task -- Different NB distributions
    ber = BernoulliNB()
    
    # NB training 
    ber.fit(shuffBowMat[:amount], shuffFv[:amount])
    
    # NB testing  
    predBer = ber.predict(shuffBowMat[:amount])

    # NB scores
    scoreBer = ber.score(shuffBowMat[amount:], shuffFv[amount:])
    
    # and another one
    mNb = MultinomialNB()
    
    # NB training 
    mNb.fit(shuffBowMat[:amount], shuffFv[:amount])
    
    # NB testing  
    predMnb = mNb.predict(shuffBowMat[:amount])

    # NB scores
    scoreMnb = mNb.score(shuffBowMat[amount:], shuffFv[amount:])
