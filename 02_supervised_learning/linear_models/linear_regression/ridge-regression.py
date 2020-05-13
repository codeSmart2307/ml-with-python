from preamble import *

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# Ridge regression with extended boston dataset
# -----------------------------------------------------------

X, y = mglearn.datasets.load_extended_boston()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

ridge = Ridge().fit(X_train, y_train)

# See training set performance
print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))

"""
Ridge is a more restricted model therefore it is less complex. This is the reason
for the low training performance but better generalization (higher test performance)
"""

# -----------------------------------------------------------
# Ridge regression with high alpha on extended boston dataset
# -----------------------------------------------------------

ridge10 = Ridge(alpha=10).fit(X_train, y_train)

# See training set performance
print("Training set score: {:.2f}".format(ridge10.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.2f}".format(ridge10.score(X_test, y_test)))

# -----------------------------------------------------------
# Ridge regression with low alpha on extended boston dataset
# -----------------------------------------------------------

ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)

# See training set performance
print("Training set score: {:.2f}".format(ridge01.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.2f}".format(ridge01.score(X_test, y_test)))

# -----------------------------------------------------------
# Visualizing coefficient magnitude with different values of alpha
# -----------------------------------------------------------

plt.plot(ridge.coef_, 's', label="Ridge alpha=1")
plt.plot(ridge10.coef_, '^', label="Ridge alpha=10")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")

plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-25, 25)
plt.legend()
plt.show()




