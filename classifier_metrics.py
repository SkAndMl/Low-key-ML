import numpy as np
import pandas as pd
from low_key.metrics import precision_binary,recall_binary,accuracy_score,f1_score
from UsefulScripts import train_test_split

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
