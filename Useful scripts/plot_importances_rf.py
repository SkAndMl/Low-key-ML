def plot_importance_rf(df, cols=None, type_of_model="c"):
    from sklearn import ensemble
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    if cols != None:
        data = df[cols].copy()
    else:
        data = df.copy()

    if type_of_model == "c":
        rf = ensemble.RandomForestClassifier()
    else:
        rf = ensemble.RandomForestRegressor()

    rf = rf.fit(data.iloc[:, :-1], data.iloc[:, -1])
    imps = rf.feature_importances_
    idxs = np.argsort(imps)
    cols = df.columns

    sns.set_style("whitegrid")
    plt.title("FEATURE IMPORTANCES", size=20)
    plt.barh(range(len(idxs)), imps[idxs])
    plt.yticks(range(len(idxs)), [cols[i] for i in idxs])
    plt.xlabel("RANDOM FOREST IMPORTANCE")
    plt.show();