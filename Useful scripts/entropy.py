import numpy as np

def entropy(value_set):
  # node impurity measurement for decision tree
  total_num = np.sum(value_set)
  ig = 0
  smoothing_term = 10e-7
  for x in value_set:
    p = (x+smoothing_term)/total_num
    ig -= p*np.log2(p)
  return np.round(ig,3)