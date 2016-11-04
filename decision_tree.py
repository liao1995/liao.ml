#============================================================================
# Name        : DecisionTree.py
# Author      : LIAO
# Version     : 1.0 2016.10.29 
# Copyright   : copyright (c)2016, LIAO
# Description : Decision Tree Implemented By Python
#============================================================================

import numpy as np 
from scipy.stats import chi2_contingency
import math
import csv

class Node:
  """ Node of the decision tree """
  def __init__(self):
    self.fid = -1	# Index of feature used for splitting
    self.feature = -1	# Feature value used for splitting the node 
    self.nodes = None	# Children of this node 
    self.parent = None	# Parent of this node
    self.label = -1	# Class of this node, ONLY for leaf
    self.entropy = None	# Entropy of data of this node
    self.depth = -1	# Depth of this node in tree
    self.ex_cts = None	# List: counts of each label examples


  def isleaf(self):
    return self.label != -1


class Tree:
  """ Decision Tree class """
  def __init__(self, epsilon=1e-5, prepruning=False, postpruning=False,
                     missing=None, criterion='gini'):
    """ 
      epsilon: if gain less than this value, see as leaf
      criterion: attribute splitting criterion, default is 'gini'
                 'gini': splitting by gini index of attribute
                 'info gain': splitting by information gain of attribute
                 'mis error': splitting by misclassification error
      prepruning: bool value, decide whether use pre-pruning method or not
      postpruning: bool value, decide whether use post-pruning method or not
      missing: missing value, algorithm will see attribute hold this 
               value as missing value
    """ 
    self.__root = None		# Root of decision tree
    self.n_classes = -1		# Number of classes for classifier
    self.epsilon = epsilon
    self.prepruning = prepruning
    self.postpruning = postpruning
    self.alpha = 0.05		# Alpha of chi-square test for pre-pruning
    self.criterion = criterion	# Attribute Splitting criterion
    self.missing = missing	# Missing value

    
  def fit(self, train_data, train_label):
    """ Fit this tree with train data and train label 
      Parameters:
      ----------
      train_data: training data, 2D array-like 
      train_label: training label, 1D array-like, for classifier,
                  class label must be value from 0 to n_classes-1 
    """  
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_data_label = np.column_stack((train_data, train_label))
    self.n_classes = np.max(train_label) + 1
    w = np.ones(train_data.shape[0])
    self.__attrs = list()	# all values of all attributes
    self.__del_attrs = set()	# store all deleted attributes
    self.n_features = train_data.shape[1]
    for i in range(self.n_features):
      self.__attrs.append(get_value_counts(train_data[:,i]))
    self.__root = self.__build_tree(train_data_label, 
                  range(train_data.shape[1]), w = w)
    if self.postpruning: self.__post_pruning()


  def __build_tree(self, data, attr_ids, depth=0, feature=None,
                   w=None):
    """ Build decision tree recursively 
        Parameters:
        -----------
        node: current node need to generate its children
        data: examples with training data and label (last column)
        attrs: attributes set, included current attributes index
        depth: Depth of current node in tree
        feature: feature value used for splitting the node
        w: weight of samples, will used for missing value case
        ----------
        Return:
        Constructed node
    """
    node = Node()
    node.feature = feature
    node.depth = depth
    if self.postpruning: 	# get counts of each label value
      node.ex_cts=np.bincount(data[:,-1],minlength=self.n_classes)
    # Empty attributes
    if len(attr_ids) == len(self.__del_attrs): 
      node.label = self.__get_max_class(data[:,-1])
      return node
    # All instances has same labels
    if np.sum(data[:,-1]==data[0][-1])==len(data[:,-1]):
      node.label = data[0][-1]
      return node
    if self.criterion == 'info gain' and self.missing==None:
      node.entropy = calc_entropy(data[:,-1])
    # Calculate criterions
    criterions = list()
    r_lists = list()
    for aid in attr_ids:
      if aid in self.__del_attrs: 
        if self.missing != None: r_lists.append(float('nan'))
        criterions.append(float('nan'))
        continue
      if self.criterion == 'info gain':
        if self.missing != None:
          if sum(data[:,aid]==self.missing)==len(data):
            continue
        info_gain, r = calc_info_gain(data[:,aid], data[:,-1],
                     node.entropy, missing=self.missing, w=w)
        criterions.append(info_gain) 
        r_lists.append(r)
      elif self.criterion == 'gini':
        criterions.append(calc_gini_index(data[:,aid], data[:,-1]))
      elif self.criterion == 'mis error':
        criterions.append(calc_mis_error(data[:,aid], data[:,-1]))
    # Threshold
    if len(criterions) < 1: 
      node.label = self.__get_max_class(data[:,-1])
      return node
    if self.criterion == 'info gain':
      max_criterions = np.nanmax(criterions)	# maximalize information gain
      if max_criterions < self.epsilon:
        node.label = self.__get_max_class(data[:,-1])
        return node
    elif self.criterion == 'gini' or self.criterion == 'mis error':
      max_criterions = np.nanmin(criterions)	
    max_index = criterions.index(max_criterions)
    if self.missing != None: max_r = r_lists[max_index]
    max_aid = attr_ids[max_index]
    self.__del_attrs.add(max_aid)
    if np.sum(data[:,max_aid]==data[0][max_aid]) == len(data[:,max_aid]):
      node.label = self.__get_max_class(data[:,-1])
      return node
    # pre-pruning
    if self.prepruning and self.__pre_pruning(data[:,max_aid], data[:,-1]):
      node.label = self.__get_max_class(data[:,-1])
      return node
    node.fid = max_aid		# split by this feature
    #del attr_ids[max_index]	# remove this attribute from attribute set
    all_attrs = self.__attrs[max_aid]
    node.nodes = list()
    for each_attr in all_attrs:	# select column max_ig_aid value is each_attr,
      if sum(data[:,max_aid]==each_attr) == 0:
        child = Node()
        child.label = self.__get_max_class(data[:,-1])
        child.feature = each_attr
        child.depth = depth + 1
        child.ex_cts = node.ex_cts
      else:			# build subtree recursively
        if self.missing != None:
          if each_attr == self.missing: continue
          new_data = list()
          new_w = list()
          for i in range(len(data)):
            if data[i][max_aid] == each_attr:
              new_data.append(data[i])
              new_w.append(w[i])
            elif data[i][max_aid] == self.missing:
              new_data.append(data[i])
              new_w.append(w[i]*float(max_r[each_attr]))
          new_data = np.array(new_data)
          new_w = np.array(new_w)
          child = self.__build_tree(new_data, attr_ids, depth=depth+1,
                                    feature=each_attr,w=new_w)
        else:
          child = self.__build_tree(data[data[:,max_aid]==each_attr,:],
                   attr_ids, depth=depth+1, feature=each_attr)
      child.parent = node
      node.nodes.append(child)
    return node
 
   
  def predict(self, test_data):
    """ Predict the label of test data, 
        return the 1D array with each element stands for test sample class """
    if self.__root == None: 
      raise Exception('You must call fit() to train first.')   
    test_data = np.array(test_data)
    labels = list()
    for sample in test_data:	# predict labels of all test samples
      node = self.__root
      while not node.isleaf():
        find = False
        for child in node.nodes:
          if sample[node.fid] == child.feature:
            node = child 	# go down to leaf
            find = True
            break
        if not find: node = node.nodes[0]
      labels.append(node.label)
    return np.array(labels)      

  
  def __pre_pruning(self, attrs, labels): 
    """ Do pre-pruning based on chi-square test """
    observe = list()
    m = get_value_counts(attrs)
    for attr in m:
      observe.append(np.bincount(labels[attrs==attr], minlength=self.n_classes))   
    try:    
      chi2, p, dof, ex = chi2_contingency(np.array(observe))
    except ValueError:
      return False
    if p < self.alpha: return False
    else: return True
 
  
  def __post_pruning(self):
    """ Do post-pruning based on Pessimistic Error Pruning """
    queue = [self.__root]
    while len(queue):
      node = queue[0]
      if node.isleaf() or np.sum(node.ex_cts)==0: continue
      del queue[0]
      et = np.sum(node.ex_cts) - np.max(node.ex_cts)
      leaves = self.get_subtree_leaves(node)
      eT = len(leaves) / 2.0
      for leaf in leaves:
        eT += np.sum(leaf.ex_cts) - np.max(leaf.ex_cts)
      Nt = np.sum(node.ex_cts)
      seT = math.sqrt(eT*(Nt-eT)/float(Nt))
      if et <= eT + seT:			# pessimistic error pruning
        node.nodes = None
        node.label = np.argmax(node.ex_cts)
      else: queue.extend(node.nodes)	# add children of node to queue
      
    
  def get_subtree_leaves(self, subtree=None):
    """ Get all leaves of this tree, return a list of leaves 
        default subtree is root of this tree """
    leaves = list()
    if subtree == None: subtree = self.__root
    queue = [subtree]
    while len(queue):
      node = queue[0]
      del queue[0]
      if node.isleaf(): leaves.append(node)
      else: queue.extend(node.nodes)
    return leaves


  def __get_max_class(self, train_label):
    """ Get the maximum number of class label """
    num_labels = list()
    for i in range(self.n_classes):
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
        if node.isleaf(): 
          print('leaf - by.{0}, c.{1}'.format(node.feature, node.label))
        else: print('by.{0} s.{1}\t'.format(node.feature, node.fid))
        if node.nodes != None: q.append(node.nodes)
   

class AdaBoost:
  """ AdaBoost class """
  def __init__(self,  n_estimators=10, prepruning=False, 
               postpruning=False, missing=None, criterion='gini',
               verbose=False):
    """ 
      n_estimators: number of base estimators (decision tree)
      criterion: attribute splitting criterion, default is 'gini'
                 'gini': splitting by gini index of attribute
                 'info gain': splitting by information gain of attribute
                 'mis error': splitting by misclassification error
      prepruning: bool value, decide whether use pre-pruning method or not
      postpruning: bool value, decide whether use post-pruning method or not
      missing: missing value, algorithm will see attribute hold this 
               value as missing value
      verbose: print out detail or not
    """ 
    self.n_estimators = n_estimators
    self.prepruning = prepruning
    self.postpruning = postpruning
    self.missing = missing
    self.criterion = criterion
    self.n_classes = -1
    self.verbose = verbose
    

  def fit(self, train_data, train_label):
    """ Fit this tree with train data and train label 
      Parameters:
      ----------
      train_data: training data, 2D array-like 
      train_label: training label, 1D array-like, for classifier,
                  class label must be value from 0 to n_classes-1 
    """      
    m = len(train_data)	# m: number of samples
    self.n_classes = np.max(train_label) + 1
    train_data_label = np.column_stack((train_data, train_label))
    D = np.ones(m) / m	
    self.__estimators = list()
    self.__alphas = list() 
    for i in range(self.n_estimators):
      data = self.__sampling(train_data_label, D)
      est = Tree(prepruning=self.prepruning,
                 missing=self.missing,
                 postpruning=self.postpruning, 
                 criterion = self.criterion)
      est.fit(data[:,:-1], data[:,-1])
      self.__estimators.append(est)
      y_pre = est.predict(data[:,:-1])
      e = sum(D[y_pre!=data[:,-1]])
      if e > 0.5: break     
      alpha = np.log((1 - e)/float(e+1e-20)) / 2.0 
      self.__alphas.append(alpha)
      D = np.exp((1 - (y_pre==data[:,-1]) * 2) * alpha) * D
      D /= sum(D)	
      if self.verbose: print ('Done Adaboost Round ' + str(i+1), ' e:', e, ' alpha:', alpha)     


  def predict(self, test_data):
    """ Do test and return the class labels """
    pred_matrix = list()
    for est in self.__estimators:
      pred_matrix.append(est.predict(test_data))
    pred_matrix = np.array(pred_matrix).transpose()
    cls_matrix = list()
    for cls in range(self.n_classes):
      cls_matrix.append(np.sum(self.__alphas*(pred_matrix==cls),axis=1))
    cls_matrix = np.array(cls_matrix).transpose()
    return np.argmax(cls_matrix, axis=1)
 

  def __sampling(self, data, w):
    """ Do sample on data based on weight vector w """
    n_samples = len(data)
    index = n_samples - sum(np.cumsum(w)[:,None] > 
                            np.random.rand(n_samples))
    return data[index,:]


def calc_entropy(vector, w=None):
  """ Calcualte the entropy on giving vector
      if a non-None value has been assigned to w, then 
      weighted entropy will be calculated.
  """
  if np.sum(vector==vector[0])==len(vector): return 0.0
  if w != None and len(w) != len(vector):
    raise ValueError('Length must be identical')
  n_samples = len(vector) 
  m = get_value_counts(vector)
  entropy = 0.0
  if w != None: sum_weigts = sum(w) 
  for ci in m:
    if w == None: pi = float(m[ci]) / n_samples
    else:
      pi = float(sum(w[vector==ci])) / sum_weigts 
    entropy -= pi * math.log(pi, 2)
  return entropy


def get_value_counts(vector):
  """ Get counts of all values in vector """
  m = dict()
  for value in vector:
    if value in m: m[value] += 1
    else: m[value] = 1
  return m


def calc_info_gain(attrs, labels, p_entropy=None, 
                   missing=None, w=None):
  """ Calculate the information gain of attributes
      attrs, giving the label labels corresponding 
      If giving paramter p_entropy , will using this
      parent entropy instead of calculating it. """
  if missing != None: 
    lines_no_missing = (attrs != missing)
    attrs_no_missing = attrs[lines_no_missing]
    labels_no_missing = labels[lines_no_missing]
    w_no_missing = w[lines_no_missing]
    p_entropy = calc_entropy(labels_no_missing, w_no_missing)
    info_gain = p_entropy
    m_attrs = get_value_counts(attrs_no_missing)
    sum_weights_no_missing = sum(w_no_missing)
    rm = dict()
    for a in m_attrs:
      ent_a = calc_entropy(labels_no_missing[attrs_no_missing==a], 
                           w_no_missing[attrs_no_missing==a])
      r = sum(w_no_missing[attrs_no_missing==a])/sum_weights_no_missing
      rm[a] = r
      info_gain -= r * ent_a 
    rho = sum_weights_no_missing / float(sum(w))
    return rho * info_gain, rm 
  # no missing values
  if p_entropy == None: p_entropy = calc_entropy(labels)
  info_gain = p_entropy
  m = get_value_counts(attrs)
  N = len(attrs)
  for attr in m:
    info_gain -= m[attr] / float(N) * calc_entropy(labels[attrs==attr]) 
  return info_gain, None 


def calc_gini(vector):
  """ Calculate the gini on giving vector """
  if np.sum(vector==vector[0])==len(vector): return 0.0
  p = np.bincount(vector) / float(len(vector))
  return 1.0 - sum(p ** 2)
  

def calc_gini_index(attrs, labels):
  """ Calculate the Gini index of attributes attrs,
      giving the label labels corresponding
  """
  m = get_value_counts(attrs)
  N = len(attrs)
  gini_index = 0.0
  for attr in m:
    gini_index += (m[attr] / float(N)) * calc_gini(labels[attrs==attr]) 
  return gini_index


def calc_mis(vector):
  """ Calculate the classification error on giving vector """
  if np.sum(vector==vector[0])==len(vector): return 0.0
  p = np.bincount(vector) / float(len(vector))
  return 1.0 - np.max(p)


def calc_mis_error(attrs, labels):  
  """ Calculate the misclassification error of attributes attrs,
      giving the label labels corresponding
  """
  m = get_value_counts(attrs)
  N = len(attrs)
  mis_error = 0.0
  for attr in m:
    mis_error += (m[attr] / float(N)) * calc_mis(labels[attrs==attr]) 
  return mis_error


def evaluate(model, X,  y, method='cv', times=1000, verbose=False):
  """ Evaluate giving model based on train data and train label
      Paramter:
      ---------
      model: model need train and evaluate,  must have method
             fit (train model) and predict (test) 
      X: giving training data 
      y: training label corresponding to train data 
      method: 'cv': 10-fold cross validation
              'hold-out': hold out method 
              'bootstrap': bootstrap method
      times: how many times evalution should do to calculate
             average accuracy for hold-out method
      verbose: print solving process message or not
      ---------
      Return:
      Accuracy of predict label    
  """
  y_counts = np.bincount(y)
  data_label = np.column_stack((X, y))
  if method == 'hold-out':	# level-sampling, do 1/3 hold-out
    sum_accuracy = 0.0
    for i in range(times):
      train_data_label, test_data_label = hold_out_split(X, y, y_counts) 
      model.fit(train_data_label[:,:-1], train_data_label[:,-1])
      accu = accuracy(test_data_label[:,-1], 
                      model.predict(test_data_label[:,:-1]))
      sum_accuracy += accu
      if verbose: print ('{0}\t{1}'.format(i, accu))
    return sum_accuracy / times
  if method == 'cv':		# 10-fold cross validation
    sum_accuracy = 0.0
    for i in range(times):
      np.random.shuffle(data_label)
      n_samples = len(y)
      test_size = n_samples / 10
      local_sum_accuracy = 0.0
      for j in range(10): 
        start = j * test_size
        if j == 10: end = n_samples
        else: end = (j + 1) * test_size
        test_data_label = data_label[start:end,:]
        train_data_label = np.delete(data_label, range(start,end), axis=0)
        model.fit(train_data_label[:,:-1], train_data_label[:,-1])
        accu = accuracy(test_data_label[:,-1], 
                        model.predict(test_data_label[:,:-1]))
        local_sum_accuracy += accu 
      local_sum_accuracy /= 10.0	# accuracy of 10-fold cv
      if verbose: print ('{0}\t{1}'.format(i, local_sum_accuracy))
      sum_accuracy += local_sum_accuracy
    return sum_accuracy / times
  if method == 'bootstrap':		# .632 bootstrap method
    sum_accuracy = 0.0
    for i in range(times):
      train_data_label, test_data_label = bootstrap_sampling(X, y)
      model.fit(train_data_label[:,:-1], train_data_label[:,-1])
      sample_accu = accuracy(test_data_label[:,-1], 
                      model.predict(test_data_label[:,:-1]))
      total_accu = accuracy(data_label[:,-1],
                            model.predict(data_label[:,:-1]))
      accu = (0.632 * sample_accu + 0.368 * total_accu)
      sum_accuracy += accu
      if verbose: print ('{0}\t{1}'.format(i, accu))
    return sum_accuracy / times 
  raise ValueError('Unexcepted evaluate method: ', method)


def hold_out_split(X, y, y_counts=None):
  """ 1/3 Hold out method to split the data 
      Parameter:
      ----------
      X: training data that need to be splited
      y: training label corresponding to train data
      y_counts: count of all classes of labels
      ----------
      Return:
      splited train data and label array and test data and label array
  """
  if y_counts == None: y_counts = np.bincount(y) 
  y_counts = y_counts / 3
  data_label = np.column_stack((X, y))
  np.random.shuffle(data_label)
  test_lines = list()
  train_lines = list()
  for i in range(len(data_label)):
    if y_counts[data_label[i][-1]] > 0:
      test_lines.append(i)
      y_counts[data_label[i][-1]] -= 1
    else:
      train_lines.append(i)
  return data_label[train_lines,:], data_label[test_lines,:]


def bootstrap_sampling(X, y):
  """ Sampling using bootstrap method 
      Return the bootstrap sampling result which has same size
        with [X|y], and the rest samples which has not been selected
  """
  data_label = np.column_stack((X, y))
  n_samples = len(y)
  samples = np.random.randint(n_samples,size=n_samples)
  return data_label[samples, :], np.delete(data_label, samples, axis=0)


def accuracy(y, pred_y):
  """ Calculate the accuracy between y and predict y
      that is, ratio predicted true on total samples
  """
  return float(sum(pred_y==y))/len(y)


def rand_missing(data, fraction=0, missing=-99999):
  """ Random generate the dataset that have missing fraction * N 
      (number of elements of original dataset values), return 
      the new missing dataset, default return the copy of original 
      dataset  
      Parameter:
      -----------
      data: original dataset, array-like
      fraction: float value from 0 to 1, indicate the probability 
                to select element set to missing value
      missing: what value should be filled as missing value 
  """
  data_cp = data.copy()
  count_missing = fraction * data_cp.size
  row, col = data_cp.shape
  s = set()	# for generating N * f different couples
  while (len(s)) < count_missing:
    i = np.random.randint(row)
    j = np.random.randint(col)
    data_cp[i][j] = missing
    s.add((i,j))# added actually when (i,j) not in set
  return data_cp


if __name__ == '__main__':
#  train_data = np.array([[0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],
#                         [0,0,1,1,0,0,0,1,0,0,0,0,1,1,0],
#                         [0,0,0,1,0,0,0,1,1,1,1,1,0,0,0],
#                         [0,1,1,0,0,0,1,1,2,2,2,1,1,2,0],
#                         ]).transpose()
#  train_label = np.array([0,0,1,1,0,0,0,1,1,1,1,1,1,1,0])
#  train_data = np.array([[0,1,1,0,1,0,2,1,2,0,0,2,1,1,2,2,0],
#                         [0,0,0,1,1,2,1,1,0,0,0,0,1,1,2,0,1],
#                         [0,1,0,0,0,2,1,0,0,1,1,0,0,1,2,0,0],
#                         [0,0,0,0,1,0,1,0,2,1,0,0,0,1,2,2,1],
#                         [0,0,0,1,1,2,0,1,2,1,0,0,1,1,2,2,0],
#                         [0,0,0,1,1,1,0,1,0,0,0,0,0,0,0,1,0],
#                        ]).transpose()
#  train_label = np.array([1,1,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0])
#  t = Tree()
#  t.fit(train_data, train_label)
#  t.print_tree()
#  t = AdaBoost(postpruning=False, criterion='info gain',
#            missing=None)
#  t.fit(train_data, train_label)
#  t.print_tree()                           
#  print(evaluate(t, train_data, train_label, 'bootstrap', verbose=True))
   train_data_label = list()               
   with open('connect-4.data', 'rb') as f:
     r = csv.reader(f)
     for line in r: train_data_label.append(line)
   train_data_label = np.array(train_data_label)
   train_data_label[train_data_label=='x'] = 1
   train_data_label[train_data_label=='o'] = 2
   train_data_label[train_data_label=='b'] = 0
   train_data_label[train_data_label=='win'] = 1
   train_data_label[train_data_label=='loss'] = 2
   train_data_label[train_data_label=='draw'] = 0
   train_data_label = train_data_label.astype(int)
   print (train_data_label)
   print (train_data_label.shape)
   t = AdaBoost(prepruning=False, postpruning=True, missing=None, verbose=1)
   print ('train model now...')
#   t.fit(train_data_label[:,:-1], train_data_label[:,-1])
#   test_data = np.array([[2,0,0,0,0,0,2,1,0,0,0,0,2,1,1,0,0,0,
#                       1,1,1,0,0,0,1,2,2,0,0,0,2,0,0,0,0,0,2,0,0,0,0,0]])
#   label = t.predict(test_data)
   print (evaluate(t, train_data_label[:,:-1], train_data_label[:,-1],
          method='hold-out', times=10,verbose=1))
