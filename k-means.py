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
  for k in range(2,11):
    model = KMeansCluster(verbose=1, k=k)
    model.fit(data)
    print(model.calc_sse())
