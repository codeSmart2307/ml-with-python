from preamble import *

# --------------------------------------------------------
# forge dataset

# Synthetic two-class classification dataset
# Contains 26 data points and 2 features
# --------------------------------------------------------

# generate dataset
X, y = mglearn.datasets.make_forge()
# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X.shape:", X.shape)

# Show plot
# plt.show()

# --------------------------------------------------------
# wave dataset

# Synthetic regression dataset
# Contains 1 input feature and 1 continuous target variable (response)
# --------------------------------------------------------

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel("Feature")
plt.ylabel("Target")

# Show plot
# plt.show()

# --------------------------------------------------------
# Wisconsin Breast Cancer dataset

# Records clinical measurements of breast cancer tumours
# Real-world classification dataset
# Contains 569 data points (212 malignant/357 benign) and 30 features
# --------------------------------------------------------

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys():\n", cancer.keys())
print("Shape of cancer data:", cancer.data.shape)
print("Sample counts per class:\n", 
    {n: v for n, v in zip(cancer.target_names, 
                        np.bincount(cancer.target))})
print("Feature names:\n", cancer.feature_names)

# --------------------------------------------------------
# Boston Housing dataset

# Records attributes of homes in Boston neighborhoods in the 1970s
# Real-world regression dataset
# Contains 506 data points and 13 features
# --------------------------------------------------------

from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape:", boston.data.shape)

# --------------------------------------------------------
# Extended Boston Housing dataset

# Real-world regression dataset
# Contains 506 data points and 104 features (13 original features 
# combined with 91 possible combinations of two features within those 13)
# --------------------------------------------------------

X, y = mglearn.datasets.load_extended_boston()
print("X.shape:", X.shape)