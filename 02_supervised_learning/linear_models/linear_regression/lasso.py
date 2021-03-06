from preamble import *

from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# Lasso regression with extended boston dataset
# -----------------------------------------------------------

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

lasso = Lasso().fit(X_train, y_train)

# See training set performance
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
# See number of features used
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))

"""
Here, both the training and test performance is quite bad. This means we are underfitting.
We have also only utilized 4 of the 105 features.
"""

# -----------------------------------------------------------
# Lasso regression with lower alpha on extended boston dataset
# -----------------------------------------------------------

# Reduce alpha and increase max_iter to avoid warnings
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train) 

# See training set performance
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test)))
# See number of features used
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

"""
If we reduce alpha further, we run the risk of removing the effect of regularization
and overfitting the model (see below).
"""

# -----------------------------------------------------------
# Lasso regression with even lower alpha on extended boston dataset
# -----------------------------------------------------------

# Reduce alpha and increase max_iter to avoid warnings
lasso0001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train) 

# See training set performance
print("Training set score: {:.2f}".format(lasso0001.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.2f}".format(lasso0001.score(X_test, y_test)))
# See number of features used
print("Number of features used: {}".format(np.sum(lasso0001.coef_ != 0)))

# -----------------------------------------------------------
# Visualizing coefficient magnitude with different values of alpha
# -----------------------------------------------------------

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

plt.plot(lasso.coef_, 's', label="Lasso alpha=1") 
plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01") 
plt.plot(lasso0001.coef_, 'v', label="Lasso alpha=0.0001")

plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1") 
plt.legend(ncol=2, loc=(0, 1.05))
plt.ylim(-25, 25)
plt.xlabel("Coefficient index") 
plt.ylabel("Coefficient magnitude")
plt.show()



















