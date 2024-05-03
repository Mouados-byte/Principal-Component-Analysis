from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

class PCA:
  def __init__(self, n_components = 2):
    self.n_components = n_components
    self.components = None
    self.mean = None

  def fit(self, X):
    self.X = X
    # Compute the mean of the input data
    self.mean = np.mean(X, axis=0)
    self.std = np.std(X, axis=0)

    # Center the data by subtracting the mean
    X_centered_reduced = self.center_reduce(X)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_centered_reduced.T)

    # Compute the eigenvalues and eigenvectors of the covariance matrix
    self.eigenvalues, self.eigenvectors = np.linalg.eig(cov_matrix)

    # Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(self.eigenvalues)[::-1]
    sorted_eigenvalues = self.eigenvalues[sorted_indices]
    sorted_eigenvectors = self.eigenvectors[:, sorted_indices]

    # Select the top n_components eigenvectors
    self.components = sorted_eigenvectors[:, :self.n_components]

  def transform(self, X):
    # Center the data by subtracting the mean and deviding by the standard deviation
    X_centered_reduced = self.center_reduce(X)

    # Project the data onto the principal components
    X_transformed = np.dot(X_centered_reduced, self.components)

    return X_transformed
  
  def fit_transform(self, X):
    self.fit(X)
    return self.transform(X)
  
  def center(self, X):
    self.mean = np.mean(X, axis=0)
    return X - self.mean
  
  def center_reduce(self, X):
    self.std = np.std(X, axis=0)
    return self.center(X) / self.std
  
  def visualize_2d(self, X, y):
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)

    targets = np.unique(y)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for target, color in zip(targets, colors):
        indicesToKeep = y == target
        ax.scatter(X.loc[indicesToKeep, 'principal component 1']
                  , X.loc[indicesToKeep, 'principal component 2']
                  , c = color
                  , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()
    
  def explained_variance_ratio_(self):
    # Calculate total sum of eigenvalues
    total_eigenvalues_sum = np.sum(self.eigenvalues)
    
    # Calculate explained variance ratio for each component
    return (self.eigenvalues / total_eigenvalues_sum)[:self.n_components]
    

