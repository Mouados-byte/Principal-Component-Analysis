import pandas as pd
from pca import PCA 

# Data source uri
data_source = "./datasets/iris.csv"

# load dataset into Pandas DataFrame
X = pd.read_csv(data_source, names=['sepal length','sepal width','petal length','petal width','target'])

# Features
features = ['sepal length', 'sepal width', 'petal length', 'petal width']

# Separating out the features
x = X.loc[:, features].values

# Instancing my PCA class
pca = PCA(n_components=2)

# Get the principal components of the data
principalComponents = pca.fit_transform(x)

# Create a new DataFrame with the principal components
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component {}'.format(i+1) for i in range(2)])

# Concatenate the principal components with the target column specific to each row
finalDf = pd.concat([principalDf, X[['target']]], axis = 1)

# Print the value maintained in each principal component
print(pca.explained_variance_ratio_())

# Visualize the data in 2D
pca.visualize_2d(finalDf, finalDf['target'])