def base_models_cv(models,param_grids,X,y,cv=5):

    """
    :param models: list of estimators
    :param param_grids: list of dictionaries of params for each model
    :param X: training instances with their features
    :param y: target labels
    :param cv: number of folds for cross validation
    :return: prints the best params and scores for each estimator
    """

    from sklearn.model_selection import GridSearchCV

    for i,model in enumerate(models):
        gs = GridSearchCV(estimator=model,param_grid=param_grids[i],cv=cv)
        gs = gs.fit(X,y)
        print(f"Best parameters for {model}\n")
        for j in gs.best_params_.keys():
            print(f"{j:{15}}{gs.best_params_[j]}")
        print(f"Best score: {gs.best_score_}")
        print("-------------------------------")
