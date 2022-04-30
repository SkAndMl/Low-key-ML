def create_reg_folds(df, n_splits=5,target_col_name="target"):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    df = df.sample(frac=1.).reset_index(drop=True)
    num_of_bins = int(np.floor(1 + np.log2(len(df))))
    df["bins"] = pd.cut(df[target_col_name], num_of_bins, labels=False)

    st = StratifiedKFold(n_splits=n_splits)
    for fold, (tr_, val_) in enumerate(st.split(X=df, y=df["bins"])):
        df.loc[val_, "fold"] = fold

    return df.drop("bins", axis=1)