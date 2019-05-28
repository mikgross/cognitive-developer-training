# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 11:33:35 2018

@author: msahan
"""

#First Task   Importing libraries
from matplotlib import pyplot as plt 
import data_processing_functions as mfc 
from scipy.cluster.hierarchy import dendrogram, linkage 
from scipy.cluster.hierarchy import fcluster 
from sklearn.manifold import MDS 
from sklearn.metrics.pairwise import euclidean_distances 
import tfidf_bag_of_words as tfidf
from mpl_toolkits.mplot3d import Axes3D


if __name__ == "__main__":
        
    
    #Second Task   Creating Bag-of-words for clusterring 
    directory_path = '..\\0. Data\\full_dataset\\'
    #Crearing bag-of-words object
    bag_of_words = mfc.BagOfWords(directory_path,True,True,True)
    #Creating bag-of-words
    bag_of_words_matrix = bag_of_words.create_bag_of_words() 
    
    
    #Third Task   Importing clusterring method
    #Importing KMeans 
    from sklearn.cluster import KMeans 
    
    
    #Fourth Task   Creating clusterring obkect
    #Creating KMeans object
    kmeans_object=KMeans(n_clusters=4, init='k-means++', max_iter=100, random_state=3) 
    
    
    #Fifth Task   Training 
    #Model training
    kmeans_object.fit(bag_of_words_matrix)  
    
    
    #Sixth Task   Teting
    #Model testing
    kmeans_object_predict=kmeans_object.predict(bag_of_words_matrix)
    
    
    #Seventh Task   Investigating cluster centers 
    #Cluster centers 
    centers=kmeans_object.cluster_centers_ 
    
    
    #Eigth Task   Call the function that will creat Tf-Idf bag-of-words
      
    tfIdf_matrix=tfidf.TfIdfCreaor(directory_path)
    
    
    #Tenth Task   Test TfIdf Matrix on a KMeans. Do results are the same
    #Creating KMeans object
    kmeans_object_sklearn_tfidf=KMeans(n_clusters=4, init='k-means++', max_iter=100, random_state=3) 
    #Model training
    kmeans_object_sklearn_tfidf.fit(tfIdf_matrix) 
    #Model testing
    kmeans_object_sklearn_tfidf_predict=kmeans_object_sklearn_tfidf.predict(tfIdf_matrix) 

    
    #Hierarchical clustering
    #Eleventh task   generate the linkage matrix
    #linkage matrix
    Z = linkage(bag_of_words_matrix[:], 'average')    
    #max_d = 50
    #Clusterring with respect to distances     
    #clusters = fcluster(Z, max_d, criterion='distance') 
    k=2
    #Clusterring with respect to amount of clusters 
    clusters = fcluster(Z, k, criterion='maxclust') 
    
    '''
    #Twelveth task   Plot Dendrogram
    #figure size 
    plt.figure(figsize = (25, 10)) 
    #figure title
    plt.title('Hierarchical Clustering Dendrogram')
    #figure x labele
    plt.xlabel('sample index') 
    # figure y label 
    plt.ylabel('distance') 
    dendrogram(Z,                  #dendrogrma creation 
               leaf_rotation = 90.,  # rotates the x axis labels
               leaf_font_size = 8.,)  # font size for the x axis labels
    plt.show()
    '''
    
    #Thirteen Task   Results Discussion 
    #calculating similarities with the euclidean distances
    similarities = euclidean_distances(bag_of_words_matrix) 
    #MDS object 
    mds = MDS(n_components=3, dissimilarity="precomputed", random_state=1) 
    #3D representation with respect to similarities
    pos = mds.fit_transform(similarities)  
    
    first_class_indices_X = [] #empty field
    first_class_indices_Y = [] #empty field
    first_class_indices_Z = [] #empty field
    
    second_class_indices_X = [] #empty field
    second_class_indices_Y = [] #empty field
    second_class_indices_Z = [] #empty field    
    
    #Corrdinates arrays of the first and classes 
    for Counter in range(len(kmeans_object_predict)):
        #First class
        if kmeans_object_predict[Counter] == 1: 
           first_class_indices_X.append(pos[Counter,0]) #X axis
           first_class_indices_Y.append(pos[Counter,1]) #Y axis
           first_class_indices_Z.append(pos[Counter,2]) #Z axis
        else: #Second class
           second_class_indices_X.append(pos[Counter,0]) #X axis 
           second_class_indices_Y.append(pos[Counter,1]) #Y axis
           second_class_indices_Z.append(pos[Counter,2]) #Z axis
   
    fig = plt.figure() #figure 
    ax = fig.add_subplot(111, projection='3d') #3d plot 
    ax.scatter(first_class_indices_X, first_class_indices_Y, first_class_indices_Z) #Scatter plot
    ax.scatter(second_class_indices_X, second_class_indices_Y, second_class_indices_Z) #Scatter plot
      