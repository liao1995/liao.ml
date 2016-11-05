#============================================================================
# Name        : k-means.py
# Author      : LIAO
# Version     : 1.0 2016.11.4 
# Copyright   : copyright (c)2016, LIAO
# Description : K-Means Cluster Implemented By Python
#============================================================================

import numpy as np 
import math
import csv

class KMeansCluster:
  """ 
    Cluster Based on K-Means Algorithm, follow the style of sklearn,
    call fit() to train the model, call predict() test
  """
  def __init__(self, k=2, centroids=None, metric='eculidean', 
                     threshold = 1e-3, verbose=False):
    """
      Parameters:
      -----------
      k: number of clusters, 'k' in k-means
      centroids: centroids of clusters, which lenght is k
      metric: distance metric for clusterring, default is eculidean
      threshold: if moving of centroid position < threshold, then stop
      verbose: print the detail or not
    """
    self.k = k				# Number of clusters
    self.centroids = centroids		# Centroids of clusters
    self.metric = metric		# Metric of distance
    self.labels = None			# Labels of all training data
    self.threshold = threshold		# Stopping threshold
    self.max_rouds = 10000		# Max rounds ignore threshold  
    self.verbose = verbose
    self.labels = None			# Labels of train data


  def fit(self, train_data):
    """ Fit this model with train data and generate the cluster centroids 
        After calling this method, attribute self.centroids will be k-dim
        array stands for k cluster centroids
        train_data: training data, 2D array-like 
    """  
    train_data = np.array(train_data)
    self.data = train_data		# save for calculating sse
    self.n_samples = train_data.shape[0]
    # select k initial centroids randomly
    if self.centroids == None:
      self.centroids = train_data[np.random.randint(self.n_samples,size=self.k),:]
    if self.verbose: print ('Initial centroids: ', self.centroids)
    # giving all labels of training data based on distance with centroids
    self.labels = np.zeros(self.n_samples)
    for r in range(self.max_rouds):
      # update labels
      for i in range(self.n_samples):
        self.labels[i] = np.argmin(eculidean_dis(train_data[i], self.centroids))
      # update centroids
      old_centroids = self.centroids.copy()
      for i in range(len(self.centroids)):
        if self.verbose:       
          print ('[{0}] centroid: {1}  number: {2}'.format((r+1),
               self.centroids[i], np.sum(self.labels==i)))
        self.centroids[i] = np.mean(train_data[self.labels==i,:], axis=0) 
      # terminal condition
      if np.sum(eculidean_dis(old_centroids, self.centroids)) < self.threshold:
        break 
  

  def predict(self, test_data):
    """ return the label of test_data, that is label of nearest centroid """
    return np.argmin(np.array(test_data, self.centroids))


  def calc_sse(self):
    """ Calculate the sum of the squared error """
    sse = 0.0
    for i in range(len(self.centroids)):
      sse += np.sum((self.data[self.labels==i,:] - self.centroids[i]) ** 2)
    return sse

  
  def print_detail(self):
    print ('clusters:', self.clusters)


class BisectingKMeansCluster:
  """ Bisecting K-Means Cluster """
  def __init__(self, k=2, centroids=None, metric='eculidean', 
               n_estimators=10, threshold = 1e-3, verbose=False):
    """
      Parameters:
      -----------
      k: number of clusters, 'k' in k-means
      centroids: centroids of clusters, which lenght is k
      metric: distance metric for clusterring, default is eculidean
      n_estimators: number of estimators for one split
      threshold: if moving of centroid position < threshold, then stop
      verbose: print the detail or not
    """
    self.k = k
    self.centroids = centroids
    self.metric = metric
    self.threshold = threshold
    self.n_estimators = n_estimators
    self.verbose = verbose
    self.sses = None		# SSE for each cluster

  
  def fit(self, train_data):
    """ Fit this model with train data and generate the cluster centroids 
        After calling this method, attribute self.centroids will be k-dim
        array stands for k cluster centroids
        train_data: training data, 2D array-like 
    """  
    cur_k = 1				# current number of clusters
    train_data = np.array(train_data)
    n_samples = train_data.shape[0]
    self.sses = [0]
    self.data_map = dict()		# data of all clusters
    self.data_map[0] = train_data
    self.centroids = [0]

    while cur_k < self.k:
      max_id = np.argmax(self.sses)	# select maximum SSE cluster      
      models = list()
      model_sses = list()
      for i in range(self.n_estimators):
        model =  KMeansCluster(2, metric=self.metric, 
                               threshold=self.threshold)
        model.fit(self.data_map[max_id])
        models.append(model)
        model_sses.append(model.calc_sse())
      model_id = np.argmin(model_sses)
      model = models[model_id]		# least SSE model
      self.centroids[max_id] = model.centroids[0]
      self.centroids.append(model.centroids[1])
      data_0 = model.data[model.labels==0]
      data_1 = model.data[model.labels==1]
      self.data_map[max_id] = data_0
      self.data_map[cur_k] = data_1
      self.sses[max_id] = self.__calc_sse(data_0, model.centroids[0])
      self.sses.append(self.__calc_sse(data_1, model.centroids[1]))
      cur_k += 1
      if self.verbose: print (cur_k, np.array(self.centroids))
 
 
  def calc_sse(self):
    sse = 0.0
    for i in range(self.k):
      sse += self.__calc_sse(self.data_map[i], np.array(self.centroids[i]))
    return sse
   

  def __calc_sse(self, data, centroid):
    """ 
       Calculate SSE for a cluster 
       Parameters:
       -----------
       data: 2D array-like, one example each line
       centroid: 1D array-like, stand for the centroid of this cluster 
    """
    return np.sum((data - centroid) ** 2)


def eculidean_dis(m1, m2):
  """ Calculate Eculidean distance between m1 and m2 """
  return np.sqrt(np.sum((m1-m2) ** 2, axis=1))


if __name__ == '__main__':
  with open('3D_spatial_network.txt', 'rb') as f:
    reader = csv.reader(f)
    data = list()
    for row in reader:
      data.append(row[1:])	# skip the id
    data = np.array(data).astype(np.float32)
#  for k in range(2,100):
#    model = KMeansCluster(verbose=0, k=k)
#    model.fit(data)
#    print(model.calc_sse())
    model = BisectingKMeansCluster(verbose=1,k=5)
    model.fit(data)
    print (model.calc_sse())
