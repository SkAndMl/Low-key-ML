import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_silhouette(X,y):
    cluster_labels = np.unique(y)
    n_clusters = cluster_labels.shape[0]

    silhouette_vals = metrics.silhouette_samples(X,y)
    y_ticks = []
    y_ax_lower,y_ax_upper = 0,0

    for i,c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(float(i)/n_clusters)

        plt.barh(range(y_ax_lower,y_ax_upper),
                c_silhouette_vals,
                height=1.0,
                color=color,
                edgecolor="none")
        y_ax_lower += len(c_silhouette_vals)
        y_ticks.append((y_ax_lower+y_ax_upper)/2)
    
    silhouette_avg = np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,
                color="red",
                linestyle="--")
    plt.yticks(y_ticks,cluster_labels+1)
    plt.ylabel("Cluster")
    plt.xlabel("Silhoutte Coefficients")