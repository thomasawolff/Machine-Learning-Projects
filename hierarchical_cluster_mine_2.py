# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:47:06 2019

@author: moose
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

os.chdir('C:\\\\Users\\\\moose\\\\Desktop')

class clustering(object):

   # choose columns you want to cluster from dataset when instantiating class
   def __init__(self,column1,column2,file):
      self.column1 = column1
      self.column2 = column2
      self.file = file
      self.datasetNew = pd.read_csv(self.file)
      self.X = self.datasetNew.iloc[:,[self.column1,self.column2]].values
      
   def dendrogram(self,linkage):
      # using dendrogram to optimal number of clusters
      dendrogram = sch.dendrogram(sch.linkage(self.X,linkage))
      plt.title('Dendrogram')
      plt.xlabel('X-Value')
      plt.ylabel('Y-Value')
      plt.show()
      
   def agglomViz(self,affin,link,clusters):
      # fitting heiarchical clustering to the dataset
      hc = AgglomerativeClustering(n_clusters = clusters, affinity = affin, linkage = link)
      y_hc = hc.fit_predict(self.X)
      
      # visualizing data using agglomerative method
      for i in range(0,clusters):
         plt.scatter(self.X[y_hc == i, 0], self.X[y_hc == i, 1], s = 100)
      plt.show()
      plt.title('Clusters of Data: Agglomerative Method')
      plt.xlabel('X-Value')
      plt.ylabel('Y-Value')
      plt.legend()
      plt.show()
      
   def kMeansElbow(self):
      # using the elbow method to find optimal number of clusters
      wcss = []
      for i in range(1, 11):
         kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300,n_init=10,random_state=0)
         kmeans.fit(self.X)
         wcss.append(kmeans.inertia_)   
      plt.plot(range(1, 11),wcss)
      plt.title('The Elbow Method')
      plt.xlabel('Number of Data')
      plt.ylabel('WCSS')
      plt.show()
      
   def kMeansViz(self,clusterNum):
      # applying k means to the dataset
      kmeans = KMeans(n_clusters = clusterNum, init = 'k-means++',max_iter=300,n_init=10,random_state=0)
      y_kmeans = kmeans.fit_predict(self.X)
      
      # visualizing the clusters using K-means
      for i in range(0,clusterNum):
         plt.scatter(self.X[y_kmeans == i, 0], self.X[y_kmeans == i, 1], s = 100)
         plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,c='yellow',label='Centroids')
      plt.title('Clusters of Data: K-Means Method')
      plt.xlabel('X-Value')
      plt.ylabel('Y-Value')
      plt.show()


csv_file = 'mall_customers.csv'
first_column = 3
second_column = 4
number_clusters = 5
affinity = 'euclidean'
linkage = 'ward'

# create the dendrogram for hierarchical method
hc1 = clustering(first_column,second_column,csv_file)
hc1.dendrogram(linkage)

# displays the elbow method for determining custer number
elbow = clustering(first_column,second_column,csv_file)
elbow.kMeansElbow()

# create the agglomorative hierarchical cluster
hc2 = clustering(first_column,second_column,csv_file)
hc2.agglomViz(affinity,linkage,number_clusters)

# create the K-Means cluster with yellow centroids
kMeans = clustering(first_column,second_column,csv_file)
kMeans.kMeansViz(number_clusters)





