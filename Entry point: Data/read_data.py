import numpy as np

# reading in all data into a NumPy array
print('==========================')
print('= Reading wine.data file =')
print('==========================')

all_data = np.loadtxt(open("./wine.data","r"),
    delimiter=",",
    skiprows=0,
    dtype=np.float64
    )

# load class labels from column 1
y_wine = all_data[:,0]

# conversion of the class labels to integer-type array
y_wine = y_wine.astype(np.int64, copy=False)

# load the 14 features
X_wine = all_data[:,1:]

# printing some general information about the data
print('\ntotal number of samples (rows):', X_wine.shape[0])
print('total number of features (columns):', X_wine.shape[1])

# printing the 1st wine sample
float_formatter = lambda x: '{:.2f}'.format(x)
np.set_printoptions(formatter={'float_kind':float_formatter})
print('\n1st sample (i.e., 1st row):\nClass label: {:d}\n{:}\n'
      .format(int(y_wine[0]), X_wine[0]))

# printing the rel.frequency of the class labels
print('Class label frequencies')
print('Class 1 samples: {:.2%}'.format(list(y_wine).count(1)/y_wine.shape[0]))
print('Class 2 samples: {:.2%}'.format(list(y_wine).count(2)/y_wine.shape[0]))
print('Class 3 samples: {:.2%}'.format(list(y_wine).count(3)/y_wine.shape[0]))

#################
# END OF READING#
#################

##############
# Scatterplot#
##############
from matplotlib import pyplot as plt
from math import floor, ceil
from scipy.stats import pearsonr

print('================')
print('= Scatterplots =')
print('================')

plt.figure(figsize=(10,8))

for label,marker,color in zip(
        range(1,4),('x', 'o', '^'),('blue', 'red', 'green')):

    # Calculate Pearson correlation coefficient
    R = pearsonr(X_wine[:,0][y_wine == label], X_wine[:,1][y_wine == label])
    plt.scatter(x=X_wine[:,0][y_wine == label], # x-axis: feat. from col. 1
            y=X_wine[:,1][y_wine == label], # y-axis: feat. from col. 2
            marker=marker, # data point symbol for the scatter plot
            color=color,
            alpha=0.7,
            label='class {:}, R={:.2f}'.format(label, R[0]) # label for the legend
            )

plt.title('Wine Dataset')
plt.xlabel('alcohol by volume in percent')
plt.ylabel('malic acid in g/l')
plt.legend(loc='upper right')

#plt.show()

######################
# END OF SCATTERPLOTS#
######################

# Splitting into training and test dataset

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

print('========================================')
print('= Spliting dataset : training and test =')
print('========================================')

X_train, X_test, y_train, y_test = train_test_split(X_wine, y_wine,
    test_size=0.30, random_state=123)

print('Class label frequencies')

print('\nTraining Dataset:')
for l in range(1,4):
    print('Class {:} samples: {:.2%}'.format(l, list(y_train).count(l)/y_train.shape[0]))

print('\nTest Dataset:')
for l in range(1,4):
    print('Class {:} samples: {:.2%}'.format(l, list(y_test).count(l)/y_test.shape[0]))

# END OF SPLITTING

# Standardization

print('===================')
print('= Standardization =')
print('===================')

std_scale = preprocessing.StandardScaler().fit(X_train)
X_train = std_scale.transform(X_train)
X_test = std_scale.transform(X_test)

f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))

for a,x_dat, y_lab in zip(ax, (X_train, X_test), (y_train, y_test)):

    for label,marker,color in zip(
        range(1,4),('x', 'o', '^'),('blue','red','green')):

        a.scatter(x=x_dat[:,0][y_lab == label],
            y=x_dat[:,1][y_lab == label],
            marker=marker,
            color=color,
            alpha=0.7,
            label='class {}'.format(label)
            )

    a.legend(loc='upper right')

ax[0].set_title('Training Dataset')
ax[1].set_title('Test Dataset')
f.text(0.5, 0.04, 'malic acid (standardized)', ha='center', va='center')
f.text(0.08, 0.5, 'alcohol (standardized)',
    ha='center', va='center',   rotation='vertical'
    )
f.canvas.set_window_title("Standardization")

#plt.show()

# Normalization - MinMax scaling

print('==================================')
print('= Normalization - MinMax Scaling =')
print('==================================')

minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_train)
X_train_minmax = minmax_scale.transform(X_train)
X_test_minmax = minmax_scale.transform(X_test)

f, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10,5))

for a,x_dat, y_lab in zip(ax, (X_train_minmax, X_test_minmax), (y_train, y_test)):

    for label,marker,color in zip(
        range(1,4),('x', 'o', '^'),('blue','red','green')):

        a.scatter(x=x_dat[:,0][y_lab == label],
            y=x_dat[:,1][y_lab == label],
            marker=marker,
            color=color,
            alpha=0.7,
            label='class {}'.format(label)
            )

    a.legend(loc='upper left')

ax[0].set_title('Training Dataset')
ax[1].set_title('Test Dataset')
f.text(0.5, 0.04, 'malic acid (normalized)', ha='center', va='center')
f.text(0.08, 0.5, 'alcohol (normalized)', ha='center', va='center', rotation='vertical')
f.canvas.set_window_title('MinMax Scaling')

#plt.show()


# PCA
print('=================================')
print('= Principal Component Analysis  =')
print('=================================')

from sklearn.decomposition import PCA
sklearn_pca = PCA(n_components=2)
transf_pca = sklearn_pca.fit_transform(X_train)

plt.figure(figsize=(10,8))

for label,marker,color in zip(
        range(1,4),('x', 'o', '^'),('blue', 'red', 'green')):

    plt.scatter(x=transf_pca[:,0][y_train == label],
            y=transf_pca[:,1][y_train == label],
            marker=marker,
            color=color,
            alpha=0.7,
            label='class {}'.format(label)
            )

plt.xlabel('vector 1')
plt.ylabel('vector 2')

plt.legend()
plt.title('Most significant singular vectors after linear transformation via PCA')

#plt.show()

# LDA
print('=================================')
print('= Linear Discriminant Analysis  =')
print('=================================')

from sklearn.lda import LDA
sklearn_lda = LDA(n_components=2)
transf_lda = sklearn_lda.fit_transform(X_train, y_train)

plt.figure(figsize=(10,8))

for label,marker,color in zip(
        range(1,4),('x', 'o', '^'),('blue', 'red', 'green')):


    plt.scatter(x=transf_lda[:,0][y_train == label],
            y=transf_lda[:,1][y_train == label],
            marker=marker,
            color=color,
            alpha=0.7,
            label='class {}'.format(label)
            )

plt.xlabel('vector 1')
plt.ylabel('vector 2')

plt.legend()
plt.title('Most significant singular vectors after linear transformation via LDA')

#plt.show()

# LDA - Simple linear classifier
print('===================================')
print('= LDA - Simple Linear Classifier  =')
print('===================================')

# fit model
lda_clf = LDA()
lda_clf.fit(X_train, y_train)
LDA(n_components=None, priors=None)

# prediction
print('1st sample from test dataset classified as:',    lda_clf.predict(X_test[0,:]))
print('actual class label:', y_test[0])

from sklearn import metrics
pred_train_lda = lda_clf.predict(X_train)

print('Prediction accuracy for the training dataset')
print('{:.2%}'.format(metrics.accuracy_score(y_train, pred_train_lda)))

pred_test_lda = lda_clf.predict(X_test)

print('Prediction accuracy for the test dataset')
print('{:.2%}'.format(metrics.accuracy_score(y_test, pred_test_lda)))

# Confusion Matrix
print('===========================================')
print('= Confusion Matrix of the LDA-classifier  =')
print('===========================================')
print(metrics.confusion_matrix(y_test, lda_clf.predict(X_test)))

'''
# Export results
print('===============================')
print('= Exporting results to files  =')
print('===============================')
training_data = np.hstack((y_train.reshape(y_train.shape[0], 1), X_train))
test_data = np.hstack((y_test.reshape(y_test.shape[0], 1), X_test))

np.savetxt('./training_set.csv', training_data, delimiter=',')
np.savetxt('./test_set.csv', test_data, delimiter=',')
'''