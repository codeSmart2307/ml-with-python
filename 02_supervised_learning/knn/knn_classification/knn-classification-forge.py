from preamble import *

# k-NN visualization considering single neighbor
mglearn.plots.plot_knn_classification(n_neighbors=1)
# plt.show()

# k-NN visualization considering k neighbors (eg: 3)
mglearn.plots.plot_knn_classification(n_neighbors=3)
# plt.show()

# ----------------------------------------------------------
# Step 01 - Split dataset into training and test data
# ----------------------------------------------------------

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# ----------------------------------------------------------
# Step 02 - Instantiate k-NN classifier class and set 
# neighbors count
# ----------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)

# ----------------------------------------------------------
# Step 03 - Fit k-NN classifier with training set
#
# For k-NN, fitting the training set only deals with 
# storing the data
# ----------------------------------------------------------

clf.fit(X_train, y_train)

# ----------------------------------------------------------
# Step 04 - Predict labels for test set
# ----------------------------------------------------------

print("Test set predictions: {}".format(clf.predict(X_test)))

# ----------------------------------------------------------
# Step 05 - Verify model generalization (accuracy) score
# ----------------------------------------------------------

print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# ----------------------------------------------------------
# Step 06 (OPTIONAL) - Visualizing decision boundaries for 1, 3 
# and 9 nearest neighbors
#
# A decision boundary is the divide between where the algorithm 
# classifies a data point as class 0 vs. class 1 (for 2-dimensional 
# datasets)
# ----------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # The fit method returns the object self, so we can instantiate
    # and fit in one line
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbor(s)".format(n_neighbors))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axes[0].legend(loc=3)
plt.show()






