from preamble import *

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, 
    random_state=66)

training_accuracy = []
test_accuracy = []
# Try n_neighbors from 1 to 10
neighbors_settings = range(1, 11)

for n_neighbors in neighbors_settings:
    # Build model
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # Record training set accuracy
    training_accuracy.append(clf.score(X_train, y_train))
    # Record generalization accuracy
    test_accuracy.append(clf.score(X_test, y_test))

# Plot generalization
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

"""
Considering a single nearest neighbor, the prediction on the training set is perfect. 
But when more neighbors are considered, the model becomes simpler and the training 
accuracy drops. 

The test set accuracy for using a single neighbor is lower than when using more neighbors, 
indicating that using the single nearest neighbor leads to a model that is too complex. 

On the other hand, when considering 10 neighbors, the model is too simple and 
performance is even worse. The best performance is somewhere in the middle, using 
around six neighbors.
"""







