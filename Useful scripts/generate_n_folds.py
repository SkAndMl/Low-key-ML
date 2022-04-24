def generate_n_folds(data,n_splits=5):

    """
    :param data: input dataframe which should be the trainig set
    :param n_splits: number of folds required
    :return: returns the input df with an extra column "kfold" specifying the fold a particular instance belongs to
    """

    from sklearn import model_selection
    import pandas as pd
    import numpy as np

    training_data = data.copy()
    training_data[:,"kfold"] = -1*np.ones(len(data),dtype=np.int32)
    training_data = training_data.sample(frac=1)
    kf = model_selection.KFold(n_splits=n_splits)

    for fold,(trn_,val_) in enumerate(kf.split(X=training_data)):
        training_data.loc[val_,"kfold"] = fold

    return training_data