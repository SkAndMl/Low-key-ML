from sklearn import ensemble,metrics

def strat_kfold_results(models,X,y,n_splits=5,voting=True):
    skf = model_selection.StratifiedKFold(n_splits=n_splits,
                                          shuffle=True,
                                         random_state=seed)
    fold=1
    
    if voting:
        vtg_clf = ensemble.VotingClassifier([
            (f"model_{model.__class__.__name__}",model) for model in models
        ],voting="soft")
        models.append(vtg_clf)
    
    for train_ind,val_ind in skf.split(X,y):
        print(f"{'='*20}FOLD:{fold}{'='*20}")
        x_train,x_val = X[train_ind],X[val_ind]
        y_train,y_val = y[train_ind],y[val_ind]

        for model in models:
            model = model.fit(x_train,y_train)
            pred_proba = model.predict_proba(x_val)[:,1]
            roc_score = metrics.roc_auc_score(y_val,pred_proba)
            print(f"MODEL: {model.__class__.__name__} ROC SCORE: {roc_score} ")
           
        fold+=1