import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

from pca import PCA

breast = load_breast_cancer()
breast_data = breast.data
labels = np.reshape(breast.target,(569,1))
final_breast_data = np.concatenate([breast_data,labels],axis=1)

breast_dataset = pd.DataFrame(final_breast_data)
features = breast.feature_names
breast_dataset.columns = np.append(features, 'label')

breast_dataset['label'].replace(0, 'Benign',inplace=True)
breast_dataset['label'].replace(1, 'Malignant',inplace=True)

x = breast_dataset.loc[:, features].values

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(x)

# Create a new DataFrame with the principal components
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component {}'.format(i+1) for i in range(2)])

# Concatenate the principal components with the target column specific to each row
finalDf = pd.concat([principalDf, breast_dataset['label']], axis = 1)

# Print the value maintained in each principal component
print(pca.explained_variance_ratio_())

pca.visualize_2d(finalDf, finalDf['label'])
