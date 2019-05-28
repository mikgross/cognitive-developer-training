# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:06:45 2018

@author: msahan
"""

import os #importing library
import io #importing library
from sklearn.feature_extraction.text import TfidfVectorizer


def import_data(directory_path): #Data import function 
    names_of_entries = sorted(os.listdir(directory_path)) #Names of entries
    array_with_text = [] #Crearing empty array for text   
    #importing text 
    for item in names_of_entries: #Loop with for going trough all documents 
        doc_file_path=directory_path+item
        file = io.open(doc_file_path, 'r', encoding='utf-8', errors="ignore") #Open text document
        content = file.read() #Read text document 
        file.close #Close text document
        array_with_text.append(content) #Add text to the array 
    return array_with_text #Return array with text  

def TfIdfCreaor(directory_path): #Tf-Idf function  
    array_with_text=import_data(directory_path)
    bag_of_words = TfidfVectorizer(max_features=10000,
                                       stop_words='english',
                                       use_idf=True)
    tfidf_bag_of_words_matrix=bag_of_words.fit_transform(array_with_text)
    
    return tfidf_bag_of_words_matrix.toarray()