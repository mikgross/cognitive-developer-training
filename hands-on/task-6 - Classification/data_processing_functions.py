#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 12:22:37 2018

@author: mark
"""

import os 
import io
from keras.preprocessing import text
import nltk
import numpy as np
from nltk.corpus import stopwords 

#Class that creates bag-of-words with lemmatization, stopwords and frequency
# (params: directory, lemmatization, stopwords, frequency)
class BagOfWords:
    def __init__(self, file_path, lemmatization, stopwords, frequency):
        self.file_path = file_path
        self.dictionary = []
        self.lemmatization = lemmatization
        self.stopwords = stopwords
        self.frequency = frequency
        
        
    def import_data(self, file_path):
        #names of the entries
        names_of_entries = sorted(os.listdir(file_path))
        #empty array that has same size as name of entries
        array_with_text = []
        #deleting unwanted element
        if names_of_entries[0] == '.DS_Store':
            names_of_entries.remove('.DS_Store')
        #loop that importes text 
        for item in names_of_entries:
            #path to document
            doc_file_path = file_path + item
            #open file
            file = io.open(doc_file_path, 'r', encoding = 'utf-8', errors = "ignore")
            #get content
            content = file.read()
            #close file
            file.close
            array_with_text.append(content)
        #control if output size is the same as size of entries
        if len(array_with_text)>len(names_of_entries):
            return array_with_text[0:len(names_of_entries)]
        else:
            return array_with_text
    
    
    def filter_stopwords(self, array_with_tokenized_text):
        #set English stopwords
        stopset = set(stopwords.words('english'))
        #empty array for tokenized text without stopwords
        array_with_tokenized_text_without_stopwords = []
        #loop that removes all the stopwords
        for tokenized_text in array_with_tokenized_text:
            tokenized_text_stopwords = []
            for token in tokenized_text:
                #if specific word that is in text is not in a stopwords set then it is added in the array
                if token not in stopset:
                    tokenized_text_stopwords.extend([token])
            #adding text of the dcument without stopwords to the new corpus
            array_with_tokenized_text_without_stopwords.append(tokenized_text_stopwords) 
        return array_with_tokenized_text_without_stopwords
    
    
    def lemmatize(self, array_with_tokenized_text):
        #import lemmatizer
        from nltk.stem import WordNetLemmatizer
        #create a lemmatizer object
        lemmatizer = WordNetLemmatizer()
        #make an empty array for the tokenized text with lemmatization
        array_with_tokenized_text_with_lemmatization = []
        #loop that that provides lammatization
        for tokenized_text in array_with_tokenized_text:
            tokenized_lemmatized_text = []
            for token in tokenized_text:
                #if noun then lemmatize
                if lemmatizer.lemmatize(token, pos="n") == token:
                    #if verb then lemmatize
                    if lemmatizer.lemmatize(token, pos="v") == token:
                       #if adjective then lemmatize 
                       tokenized_lemmatized_text.extend([lemmatizer.lemmatize(token, pos = "a")])
                    else:
                       tokenized_lemmatized_text.extend([lemmatizer.lemmatize(token, pos = "v")])
                else: 
                   tokenized_lemmatized_text.extend([lemmatizer.lemmatize(token, pos = "n")])
            #append array with lemmatized document     
            array_with_tokenized_text_with_lemmatization.append(tokenized_lemmatized_text) 
        return array_with_tokenized_text_with_lemmatization
                
        
    def tokenize_and_filter(self, array_with_text, lemmatization, stopwords):
        #empty array for tokenized text 
        array_with_tokenized_text = []
        #loop that provides text tokenization
        for doc_text in array_with_text:
            #tokenization and filtering with keras
            tokenized_text = text.text_to_word_sequence(doc_text, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ')
            #removing non alphabetic words
            tokenized_text = [word for word in tokenized_text if word.isalpha()] 
            #adding tokonized text to array 
            array_with_tokenized_text.append(tokenized_text)
        #if lemmatization function is included then do it     
        if lemmatization == True:
            #call the lemmatize function 
            array_with_tokenized_text_with_lemmatization = BagOfWords.lemmatize(self, array_with_tokenized_text)   
            array_with_tokenized_text = array_with_tokenized_text_with_lemmatization
        #if stopwords function is included then do it 
        if stopwords == True:
            #call the stopwords function
            array_with_tokenized_text_with_stopwords = BagOfWords.filter_stopwords(self, array_with_tokenized_text)   
            array_with_tokenized_text = array_with_tokenized_text_with_stopwords
        return array_with_tokenized_text
        
    
    def create_dictionary(self, array_with_tokenized_text):
        #create emty array for dictionary 
        dictionary_array = []
        #loop that puts all words from corpus into one array
        for item in array_with_tokenized_text:
            dictionary_array.extend(item)
        #making a dictionary    
        dictionary = nltk.FreqDist(dictionary_array)
        return dictionary
    
    
    def create_bag_of_words_from_dictionary(self, dictionary, array_with_tokenized_text, frequency):
        #defining initial variables
        counter_rows = 0
        amount_of_docs = len(array_with_tokenized_text)
        #if frequency is not included then first section 
        if frequency == False:
           #defining size for the bag-of-words matrix
           bag_of_words_matrix = np.zeros((len(array_with_tokenized_text), len(dictionary)))
           #loop for filling the bag-of-words matrix 
           for document in array_with_tokenized_text:
               document_dictionary = nltk.FreqDist(document)
               counter_columns = 0
               for record in dictionary:
                   if record in document_dictionary:
                       bag_of_words_matrix[counter_rows,counter_columns] = 1
                   counter_columns += 1
               counter_rows += 1
        #if frequency is not included then first section 
        if frequency == True:
           #defining size for the bag-of-words matrix 
           bag_of_words_matrix = np.zeros((amount_of_docs,len(dictionary)), dtype=float)
           #loop for filling the bag-of-words matrix 
           for document in array_with_tokenized_text:
               document_dictionary = nltk.FreqDist(document)
               counter_columns = 0
               for record in dictionary:
                   if record in document_dictionary:
                       bag_of_words_matrix[counter_rows,counter_columns] = float(document_dictionary[record]) / float(len(document_dictionary)) * np.log(amount_of_docs / dictionary[record] + 10**(-8))               
                   counter_columns += 1 
               counter_rows += 1
           #summing over the whole rows
           sum_of_rows_bag_of_words = np.sum(bag_of_words_matrix, axis=0)
           #creating array where on the second row is summ over the whole rows and on the first will be its index
           important_words_with_indices = np.zeros((2, len(sum_of_rows_bag_of_words))) 
           #filling up second row
           important_words_with_indices[0,:] = sum_of_rows_bag_of_words
           #loop that fills up first row
           for counter in range(len(sum_of_rows_bag_of_words)):
               important_words_with_indices[1,counter] = counter
           #transposing important_words_with_indices due to technical issues with sorting
           important_words_with_indices=np.transpose(important_words_with_indices)
           #sort the array with respect to the sum
           important_words_with_indices = (sorted(important_words_with_indices, key=lambda x: x[0],reverse=True))
           #converting to array
           important_words_with_indices=np.array(important_words_with_indices)
           #transposing back
           important_words_with_indices=np.transpose(important_words_with_indices)
           #taking 10000 the most relevant terms  
           important_words_with_indices=important_words_with_indices[:,:10000]
           #if length of the bag-of-words is less than 10000 we take the whole array
           if len(bag_of_words_matrix[0]) < 10000:
              length=len(bag_of_words_matrix[0])
           else:
              length=10000
           #creating empty array for frequency bag-of-words
           freq_bag_of_words=np.zeros((len(bag_of_words_matrix), length))
           #loop that filling up frequency bag-of-words matrix with respect to the indices of the values
           for counter1 in range(len(bag_of_words_matrix)):
               for counter2 in range(length):
                   freq_bag_of_words[counter1,counter2] = bag_of_words_matrix[counter1,int(important_words_with_indices[1,counter2])]
           del bag_of_words_matrix
           bag_of_words_matrix=freq_bag_of_words 
        return bag_of_words_matrix
    
    
    def create_features_vector(self):    
        #names of entries
        names_of_entries = sorted(os.listdir(self.file_path))
        #deleting unwanted element
        if names_of_entries[0] == '.DS_Store':
            names_of_entries.remove('.DS_Store')
        #getting length of the list 
        lenth_of_list = len(names_of_entries)
        #creating array for the features vector
        features_vector = np.zeros((lenth_of_list, 1))
        counter = 0
        #loop that fills up features vector with respect to the name of the documents 
        for item in names_of_entries:
            if item[0:3] == 'alt':
                features_vector[counter,0] = 1
            if item[0:3] == 'sci':        
                features_vector[counter,0] = 2
            if item[0:4] == 'takl':
                features_vector[counter,0] = 3
            if item[0:3] == 'med':
                features_vector[counter,0] = 4
            counter += 1       
        return features_vector
    
    
    def shuffle_array(self, bag_of_words, features_vector):
        #defining array for shuffling
        array_to_shuffle = np.zeros((len(bag_of_words),len(bag_of_words[0])+1))
        #first column is a features vector
        array_to_shuffle[:,0] = np.matrix(features_vector[:,0])
        #rest of the matrix is a bag-of-words matrix
        array_to_shuffle[:,1:] = np.matrix(bag_of_words)
        #shuffling rows
        np.random.shuffle(array_to_shuffle)
        #return back shuffled features vecotor and bag-of-words matrix
        return array_to_shuffle[:,1:], array_to_shuffle[:,0]
            
                
    def create_bag_of_words(self):
        #create array with text from the whole set of documents
        array_with_text = BagOfWords.import_data(self, self.file_path)
        #tokenize the text 
        array_with_tokenized_text = BagOfWords.tokenize_and_filter(self, array_with_text, self.lemmatization, self.stopwords)
        #create the dictionary
        self.dictionary = BagOfWords.create_dictionary(self, array_with_tokenized_text)
        #create bag-of-words matrix
        bag_of_words_matrix = BagOfWords.create_bag_of_words_from_dictionary(self, self.dictionary, array_with_tokenized_text, self.frequency) 
        return bag_of_words_matrix
        
        
        
        
        