def scatter_of(column, data, hue=None, thresh=0.25):
    """
    This is used to display the scatterplots of all continuous valued cols with respect to another column.
    column -> the column with which other features have to be compared.
    data -> dataframe
    hue -> categorical column to add a 3rd dimesnion to the scatterplots.
    thresh -> this value is used to determine whether a given column is a continuous valued column or not.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")

    plot_cols = []
    for i in data.columns:
        if data[i].nunique() / len(data) >= thresh and i != column:
            plot_cols.append(i)

    fig, axes = plt.subplots(nrows=len(plot_cols), figsize=(10, 5), sharex=True)
    for i, col in enumerate(plot_cols):
        if hue is None:
            sns.scatterplot(x=column, y=col, data=data, ax=axes[i])
        else:
            sns.scatterplot(x=column, y=col, hue=hue, data=data, ax=axes[i])
    fig.tight_layout()
    plt.show()