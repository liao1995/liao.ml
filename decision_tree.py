#============================================================================
# Name        : DecisionTree.py
# Author      : LIAO
# Version     : 1.0 2016.10.29 
# Copyright   : copyright (c)2016, LIAO
# Description : Decision Tree Implemented By Python
#============================================================================

import numpy as np 
import math

class Node:
  """ Node of the decision tree """
  def __init__(self):
    self.fid = -1	# Index of feature used for splitting
    self.feature = -1	# Feature value used for splitting the node 
    self.nodes = None	# Children of this node 
    self.parent = None	# Parent of this node
    self.label = -1	# Class of this node, ONLY for leaf
    self.entropy = -1	# Entropy of data of this node
    self.depth = -1	# Depth of this node in tree

  
  def isleaf(self):
    return self.label != -1


class Tree:
  """ Decision Tree class """
  def __init__(self, epsilon=1e-5):
    self.__root = None		# Root of decision tree
    self.n_classes = -1		# Number of classes for classifier
    self.epsilon = epsilon	# If gain less than this value, see as leaf
    
  def fit(self, train_data, train_label):
    """ Fit this tree with train data and train label 
      Parameters:
      ----------
      train_data: training data, 2D array-like 
      train_label: training label, 1D array-like, for classifier,
                  class label must be value from 0 to n_classes-1 
    """  
    try:
      train_data = np.array(train_data)
      train_label = np.array(train_label)
      train_data_label = np.column_stack((train_data, train_label))
    except Exception, e:
      print(e)
      exit()
    self.n_classes = np.max(train_label) + 1
    self.__root = self.__build_tree(train_data_label, range(train_data.shape[1]))
    print(self.__root)


  def __build_tree(self, data, attr_ids, depth=0,feature=None):
    """ Build decision tree recursively 
        Parameters:
        -----------
        node: current node need to generate its children
        data: examples with training data and label (last column)
        attrs: attributes set, included current attributes index
        depth: Depth of current node in tree
        feature: feature value used for splitting the node
        ----------
        Return:
        Constructed node
    """
    node = Node()
    node.feature = feature
    node.depth = depth
    # All instances has same labels
    if np.sum(data[:,-1]==data[0][-1])==len(data[:,-1]):
      node.label = data[0][-1]
      return node
    # Empty attributes
    if len(attr_ids) == 0: 
      node.label = __get_max_class(data[:,-1])
      return node
    node.entropy = calc_entropy(train_label)
    # Calculate infomration gain
    info_gains = list()
    for aid in attr_ids:
      info_gains.append(calc_info_gain(data[:,aid], data[:,-1], node.entropy)) 
    # Threshold
    max_info_gains = max(info_gains)
    if max_info_gains < self.epsilon:
      node.label = __get_max_class(data[:,-1])
      return node
    max_ig_index = info_gains.index(max_info_gains)
    max_ig_aid = attr_ids[max_ig_index]
    node.fid = max_ig_aid	# split by this feature
    del attr_ids[max_ig_index]	# remove this attribute from attribute set
    all_attrs = get_value_counts(data[:, max_ig_aid])
    node.nodes = list()
    for each_attr in all_attrs:	# select column max_ig_aid value is each_attr,
      child = self.__build_tree(# then delete this column, pass to child node
               np.delete(data[data[:,max_ig_aid]==each_attr,:], max_ig_aid, axis=1),
               attr_ids, depth=depth+1, feature=each_attr) 
      child.parent = node
      node.nodes.append(child)
    return node
 

  def __get_max_class(self, train_label):
    """ Get the maximum number of class label """
    num_labels = list()
    for i in range(n_classes):
      num_labels.append(np.sum(train_label == i))
    return np.argmax(num_labels)

  def print_tree(self):
    """ Print the structure of this tree """
    print ('--level 0--\ns.{0}'.format(self.__root.fid))
    q = list()
    level = 0
    q.append(self.__root.nodes)
    while len(q) != 0:
      nodes = q[0]
      del q[0]
      for node in nodes:
        if node.depth > level: 
          print('--level {0}--'.format(node.depth))
          level = node.depth
        if node.isleaf(): print('leaf - by.{0}, c.{1}'.format(node.feature, node.label))
        else: print('by.{0} s.{1}\t'.format(node.feature, node.fid))
        if node.nodes != None: q.append(node.nodes)
   

def calc_entropy(vector):
  """ Calcualte the entropy on giving vector
      Noting: vector value from 0 to n_classes-1,
              just like training labels  
  """
  p = np.bincount(vector) / float(len(vector))
  entropy = 0.0
  for pi in p: 
    if pi != 0.0: entropy -= pi * math.log(pi, 2)  
  return entropy


def get_value_counts(vector):
  """ Get counts of all values in vector """
  m = dict()
  for value in vector:
    if value in m: m[value] += 1
    else: m[value] = 1
  return m


def calc_info_gain(attrs, labels, p_entropy=None):
  """ Calculate the information gain of attributes
      attrs, giving the label labels corresponding 
      If giving paramter p_entropy , will using this
      parent entropy instead of calculating it. """
  if p_entropy == None: p_entropy = calc_entropy(labels)
  info_gain = p_entropy
  m = get_value_counts(attrs)
  N = len(attrs)
  for attr in m:
    info_gain -= m[attr] / float(N) * calc_entropy(labels[attrs==attr]) 
  return info_gain 


if __name__ == '__main__':
  train_data = np.array([[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],
                         [0,0,1,1,0,0,0,1,0,0,0,0,1,1,0],
                         [0,0,0,1,0,0,0,1,1,1,1,1,0,0,0],
                         [0,1,1,0,0,0,1,1,2,2,2,1,1,2,0],
                         ]).transpose()
  train_label = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
  t = Tree()
  t.fit(train_data, train_label)
  t.print_tree()
