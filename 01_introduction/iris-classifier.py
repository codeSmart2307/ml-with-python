from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import mglearn

# ------------------------------------------------------------------------------
# Load the dataset
# ------------------------------------------------------------------------------

iris_dataset = load_iris()

# ------------------------------------------------------------------------------
# Exploring the dataset
# ------------------------------------------------------------------------------

# See keys of Bunch object (similar to dictionary)
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# See description of dataset
print(iris_dataset['DESCR'])
# See classes (species of iris flowers)
print("Target names: {}".format(iris_dataset['target_names']))
# See features
print("Feature names: \n{}".format(iris_dataset['feature_names']))
# See type of data (numeric measurements of features)
print("Type of data: {}".format(type(iris_dataset['data'])))
# See shape of data
print("Shape of data: {}".format(iris_dataset['data'].shape))
# See data (Feature values for samples (first 5))
print("First five columns of data:\n{}".format(iris_dataset['data'][:5]))
# See type of target (species of each flower measures)
print("Type of target: {}".format(type(iris_dataset['target'])))
# See shape of target
print("Shape of target: {}".format(iris_dataset['target'].shape))
# See target (encoded as integers from 0 to 2 for each species)
print("Target:\n{}".format(iris_dataset['target']))

# ------------------------------------------------------------------------------
# Creating training and test data
# ------------------------------------------------------------------------------

# Setting random_state provides the function with a fixed seed, 
# making the outcome deterministic (same output every time)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

# See shapes of training and test data
print("X_train shape: {}".format(X_train.shape)) 
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape)) 
print("y_test shape: {}".format(y_test.shape))

# ------------------------------------------------------------------------------
# Visualize data to identify abnormalities
# ------------------------------------------------------------------------------

# Create dataframe from data in X_train
# Label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# Create a scatter plot from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60,
                           alpha=.8, cmap=mglearn.cm3)

# ------------------------------------------------------------------------------
# Building the model
# ------------------------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier 
# The k-nearest neighbors algorithm is used for this model
# We will use only a single neighbor to determine the data point
knn = KNeighborsClassifier(n_neighbors=1)

# Build model on the training set
knn.fit(X_train, y_train)

# ------------------------------------------------------------------------------
# Making predictions
# ------------------------------------------------------------------------------

# New iris with sepal length 5cm, sepal width 2.9cm, petal length 1cm 
# and petal width 0.2cm.
# The measurements have been made into a single row in a 2-d Numpy
# array as scikit-learn always expects 2-d arrays for data.
X_new = np.array([[5, 2.9, 1, 0.2]]) 
print("X_new.shape: {}".format(X_new.shape))

# Predict species
prediction = knn.predict(X_new)
# See class index of predicted species
print("Prediction: {}".format(prediction)) 
# See name of predicted species
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# ------------------------------------------------------------------------------
# Evaluating the model
# ------------------------------------------------------------------------------

# Calculating accuracy of the model
# Accuracy is the fraction of flowers for which the right species was predicted
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

# Comparing test set predictions with test set labelsn and computing the score
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))

# The 2 steps above can be compressed into one by using the score() method
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))