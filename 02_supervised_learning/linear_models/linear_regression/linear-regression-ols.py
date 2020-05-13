from preamble import *

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# Linear regression with wave dataset
# -----------------------------------------------------------

# Generate wave dataset for 60 samples
X, y = mglearn.datasets.make_wave(n_samples=60)
# Split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# Fit training data to model
lr = LinearRegression().fit(X_train, y_train)

# See slope parameters (aka weights/coefficients, w)
print("lr.coef_: {}".format(lr.coef_))
# See offset (intercept, b)
print("lr.intercept_: {}".format(lr.intercept_))

# See training set performance
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

"""
This is an underfitted model due to the fact that both training and test 
R^2 scores are low and also because the dataset is low dimensional.
"""

# -----------------------------------------------------------
# Linear regression with extended boston dataset
# -----------------------------------------------------------

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

# See training set performance
print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))

"""
This is an overfitted model due to the fact that the training R^2 score 
is very high whereas the test R^2 score is low
"""
