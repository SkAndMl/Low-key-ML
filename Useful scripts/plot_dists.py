def plot_dists(data,cols,hue=None,bins=20,plot='hist'):
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt
    n_rows = int(np.ceil(len(cols)/2))
    col = 0
    sns.set_style("whitegrid")
    fig,axes = plt.subplots(nrows=n_rows,ncols=2,figsize=(20,15))
    for i in range(n_rows):
        for j in range(2):
            if col > len(cols):
                axes[i][j].axis("off")
                break
            if plot=='hist':
                sns.histplot(x=cols[col],data=data,hue=hue,ax=axes[i][j],bins=bins)
                axes[i][j].set_title(f'Distribution of {cols[col]}',size=10)
            elif plot=='count':
                sns.countplot(x=cols[col],data=data,hue=hue,ax=axes[i][j])
                axes[i][j].set_title(f'Count of {cols[col]}',size=10)
            elif plot == 'kde':
                sns.kdeplot(x=cols[col],data=data,hue=hue,ax=axes[i][j])
                axes[i][j].set_title(f'Distribution of {cols[col]}',size=10)
            fig.tight_layout()
            col += 1