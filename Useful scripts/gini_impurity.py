import numpy as np

def gini_impurity(value_set):
  # node impurity measurement for decision tree
  total_num = np.sum(value_set)
  gini = 1
  for j in value_set:
    gini -= (j/total_num)**2
  return np.round(gini,3)