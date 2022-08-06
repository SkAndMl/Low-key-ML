from sklearn import base
import pandas as pd

class WeightedEnsembleClassifier(base.BaseEstimator):

    def __init__(self,models,cv_strategy):

        super(WeightedEnsembleClassifier,self).__init__()
        if not isinstance(models,list):
            models = (models,)
        
        self.models = models
        self.cv_strategy = cv_strategy
    
    def fit(self,X,y):

        if isinstance(X,pd.DataFrame):
            X = X.values
        
        if isinstance(y,pd.DataFrame) or isinstance(y,pd.Series):
            y = y.values

        cv_score_dict = {i:[] for i in range(len(self.models))}

        for train_ind,val_ind in self.cv_strategy.split(X,y):
            X_train,X_test = X[train_ind],X[val_ind]
            y_train,y_test = y[train_ind],y[val_ind]

            for i in range(len(self.models)):
                self.models[i] = self.models[i].fit(X_train,y_train)
                pred_proba = self.models[i].predict_proba(X_test)[:,1]
                cv_score_dict[i].append(metrics.roc_auc_score(y_test,pred_proba))
        
        tot_wt = 0
        for i in range(len(self.models)):
            cv_score_dict[i] = np.array(cv_score_dict[i]).mean()
            tot_wt += cv_score_dict[i]
        
        for i in range(len(self.models)):
            cv_score_dict[i] /= tot_wt
        
        self.wts_dict = cv_score_dict
        return self
    
    def predict(self,X):

        preds = np.zeros(shape=(len(X)))
        for i in range(len(self.models)):
            preds += self.wts_dict[i]*self.models[i].predict_proba(X)[:,1]
        
        return preds
