import numpy
from sklearn import preprocessing

# Matrix creation
'''
To avoid "UserWarning: The scale function assumes floating point values as input, got int64"
We need to set our data as decimal and not integer, so we add '.' for each number
'''

X = numpy.array([
    [1., -1., 2.],
    [2., 0., 0.],
    [0., 1., -1.]])

print X

print X.mean()
print X.var()

>>> print X
	[[ 1. -1.  2.]
	 [ 2.  0.  0.]
	 [ 0.  1. -1.]]
>>> print X.mean()
	0.444444444444
>>> print X.var()
	1.13580246914


# Or
print numpy.mean(X)
print numpy.var(X)

# Normalization of X using the scale function
X_scaled = preprocessing.scale(X)
print X_scaled

print X_scaled.mean()
print X_scaled.var()

>>> print X_scaled
	[[ 0.         -1.22474487  1.33630621]
	 [ 1.22474487  0.         -0.26726124]
	 [-1.22474487  1.22474487 -1.06904497]]
>>> print X_scaled.mean()
	0.0
>>> print X_scaled.var()
	1.0

# X2 : MinMax Normalization
X2 = numpy.array([
    [1., -1., 2.],
    [2., 0., 0.],
    [0., 1., -1.]
])

>>> print X2
	[[ 1. -1.  2.]
	 [ 2.  0.  0.]
	 [ 0.  1. -1.]]
>>> print X2.mean()
	0.444444444444
>>> print X2.var()
	1.13580246914

>>> X_MinMaxScaled = preprocessing.MinMaxScaler()
>>> X_MinMaxScaled = X_MinMaxScaled.fit_transform(X2)
>>> print X_MinMaxScaled
	[[ 0.5         0.          1.        ]
	 [ 1.          0.5         0.33333333]
	 [ 0.          1.          0.        ]]
	 
	 
# D. Data visualization
from sklearn import datasets
iris = datasets.load_iris()

import matplotlib.pyplot as plt
Xi = iris.data[:, :2]
Yi = iris.target
plt.scatter(Xi[:, 0], Xi[:, 1], c=Yi, cmap=plt.cm.Paired)
plt.show() 



