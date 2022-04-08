import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from metrics import precision_binary,recall_binary,accuracy_score,f1_score

def train_test_split(X,y,test_size=0.2,random_state=42):

    """
    Accepts only a dataframe or a numpy array as input.
    :param X: input data X
    :param y: input data y
    :param test_size: specifies the size of the test dataset.
    :param random_state: seed for shuffling the data
    :return: X_train,X_test,y_train,y_test
    """

    np.random.seed(random_state)
    shuffled_index = np.random.permutation(len(X))
    train_indices = shuffled_index[:int(len(X)*(1-test_size))]
    test_indices = shuffled_index[int(len(X)*test_size):]
    if type(X)==type(pd.DataFrame(data={1:[2,3]})):
        X_train,X_test,y_train,y_test = X.iloc[train_indices],X.iloc[test_indices],y.iloc[train_indices],y.iloc[test_indices]
        return X_train, X_test, y_train, y_test
    elif type(X)==type(np.array([1,2])):
        X_train,X_test,y_train,y_test = X[train_indices],X[test_indices],y[train_indices],y[test_indices]
        return X_train, X_test, y_train, y_test
    else:
        raise TypeError("Only dataframes and numpy arrays are accepted as input")

def plot_decision_boundary(classifier,X,y,resolution=0.02,markers=None,colors=None):
    """
    This is a function that is used to visualize the boundaries predicted by classifiers to classify the training data.
    This function only takes uses two features even if more than two are given.
    :param classifier: classifier model that is used to predict the labels
    :param X: training data
    :param y: training label
    :param resolution: resolution of the plot
    :param markers: markers for different classes
    :param colors: colors for different classes
    :return: a figure consisting of the boundaries for each class
    """

    if markers==None:
        markers = ['*','s','o']
    if colors==None:
        colors = ['blue','red','orange']

    x_min,x_max = X[:,0].min()-0.1,X[:,0].max()+0.1  # x-axis range
    y_min,y_max = X[:,1].min()-0.1,X[:,1].max()+0.1  # y_axis range

    xx,yy = np.meshgrid(np.arange(x_min,x_max,resolution),
                        np.arange(y_min,y_max,resolution))  # creating a 2x2 array for the figure

    classifier = classifier.fit(X,y)
    Z = classifier.predict(np.c_[np.ravel(xx),np.ravel(yy)])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z)  # the contour plot

    for i in np.unique(y):
        plt.scatter(X[y==i,0],X[y==i,1],color=colors[i],marker=markers[i],label=i)

    plt.legend()
    plt.show()

def classifiers_metrics(models,X,y,test_size=0.1,random_state=42):
    """
    :param models: a list or a numpy array consisting of classification models
    :param X: The whole feature set. It need not be split into training and test sets
    :param y: The whole true target labels.
    :param test_size: Size of the test data
    :param random_state: Specifies the random seed for splitting the dataset
    :return: returns a dataframe consisting of precision, recall, f1_score and accuracy of all the classifiers passed
    """
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
    precision_list,recall_list,accuracy_list,f1_list = [],[],[],[]

    if type(models)!=type([1,2,3]) and type(models)!=type(np.array([1,2,3])):
        raise TypeError("models should be of type list or numpy array")

    for model in models:
        model = model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        precision_list.append(precision_binary(y_test,y_pred))
        recall_list.append(recall_binary(y_test,y_pred))
        accuracy_list.append(accuracy_score(y_test,y_pred))
        f1_list.append(f1_score(y_test,y_pred))

    metric_df = pd.DataFrame(index=models,data={"Precision":precision_list,
                                                "Recall":recall_list,
                                                "Accuracy":accuracy_list,
                                                "F1 Score":f1_list})
    return metric_df

def gini_impurity(value_set):
  # node impurity measurement for decision tree
  total_num = np.sum(value_set)
  gini = 1
  for j in value_set:
    gini -= (j/total_num)**2
  return np.round(gini,3)

def entropy(value_set):
  # node impurity measurement for decision tree
  total_num = np.sum(value_set)
  ig = 0
  smoothing_term = 10e-7
  for x in value_set:
    p = (x+smoothing_term)/total_num
    ig -= p*np.log2(p)
  return np.round(ig,3)