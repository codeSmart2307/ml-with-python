from preamble import *

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------
# Logistic regression with default C on cancer dataset
# -----------------------------------------------------------

# Load dataset
cancer = load_breast_cancer()
# Split dataset to training and test data
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=42)
# Fit model
logreg = LogisticRegression().fit(X_train, y_train)

# See training set performance
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))

"""
The default value of C=1 provides good performance with 95% test and training accuracy
but since they are both very close together it is likely underfitting.
"""

# -----------------------------------------------------------
# Logistic regression with C=100 on cancer dataset
# -----------------------------------------------------------

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)

# See training set performance
print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))

# -----------------------------------------------------------
# Logistic regression with C=0.01 on cancer dataset
# -----------------------------------------------------------

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)

# See training set performance
print("Training set score: {:.3f}".format(logreg001.score(X_train, y_train)))
# See test set performance
print("Test set score: {:.3f}".format(logreg001.score(X_test, y_test)))

# -----------------------------------------------------------
# Visualize learned coefficients with varying regularization parameter values
# -----------------------------------------------------------

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()
# plt.show()

# -----------------------------------------------------------
# More interpretable model with L1 regularization
# -----------------------------------------------------------

for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, solver='liblinear', penalty="l1").fit(X_train, y_train)
    print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
          C, lr_l1.score(X_train, y_train)))
    print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(
          C, lr_l1.score(X_test, y_test)))
    plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))

plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")

plt.ylim(-5, 5)
plt.legend(loc=3)
plt.show()