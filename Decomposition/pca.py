from sklearn.base import TransformerMixin
import numpy as np

class PCA(TransformerMixin):

    def __init__(self, n_components=None):
        self.n_components = n_components  # dimension that the data has to be reduced to

    def fit(self, X):
        if self.n_components == None:  # if the number of components is not mentioned keep the same dimension as the data
            self.n_components = X.shape[1]

        self.cov_mat = np.cov(X.T)  # covariance matrix
        self.eigen_vals, eigen_vecs = np.linalg.eig(self.cov_mat)  # get the eigen values and vectors

        # forms a list of tuples in which each tuple has an eigen value and a vector
        eigen_pairs = [(self.eigen_vals[i], eigen_vecs[:, i]) for i in range(len(self.eigen_vals))]
        eigen_pairs.sort(key=lambda k: k[0],
                         reverse=True)  # sorts the list in descending order based on the eigen value
        self.explained_variance_ratio = [self.eigen_vals[i] / np.sum(self.eigen_vals) for i in
                                         range(len(self.eigen_vals))]

        self.W = np.empty(shape=(X.shape[1], self.n_components))
        for i in range(self.n_components):
            self.W[:, i] = eigen_pairs[i][1]  # gets the first n_components eigen vectors into the W matrix
        return self

    def transform(self, X):
        return X.dot(self.W)  # transforms a d-dimensional data into n_components-dimensional data.