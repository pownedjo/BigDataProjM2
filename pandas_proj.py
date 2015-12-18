import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


columns = ['Cultivar', 'Alcohol', 'Malic Acid', 'Ash', 'Alkalinity of Ash', 'Magnesium', 'Total phenol', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

columns_values = ['Alcohol', 'Malic Acid', 'Ash', 'Alkalinity of Ash', 'Magnesium', 'Total phenol', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']


## Datas Visualisation - Display a List of Lists
def visualize_datas(dataset, title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=18)
    plt.xlabel('Abscisse', fontsize=14)
    plt.ylabel('Ordonee', fontsize=14)

    for list_test in dataset:
        plt.plot(list_test, 'ro') # Nuage de points

    plt.show()  ## show graph


## SPLITING VALUES FOR EACH CULTIVATORS
def split_datas_into_cultivators(dataset):
	cultivar1 = dataset.loc[dataset['Cultivar'] == 1]
	cultivar2 = dataset.loc[dataset['Cultivar'] == 2]
	cultivar3 = dataset.loc[dataset['Cultivar'] == 3]
	
	display_dataset('Raw Dataset Visualisation', cultivar1, cultivar2, cultivar3)
	
	
def display_dataset(title, dataset1, dataset2, dataset3):
	fig = plt.figure()
	fig.suptitle(title, fontsize=18)
	
	for name in columns_values:
		plt.plot(dataset1[name], 'r')
		plt.plot(dataset2[name], 'g')
		plt.plot(dataset3[name], 'b')
	
	#for r in columns_values:
	#	ax = plt.subplot(cultivar1[r], cultivar2[r], cultivar3[r])
	#	fig.add_subplot(ax)
	
	plt.show()
		
	#s = cultivar1['Alcohol']
	#print s
	
	

### MAIN ###
raw_dataset = pd.read_csv(open('wine.txt'), names=columns)

#description_of_dataset = raw_dataset.describe()
#alcohol_values = raw_dataset['Alcohol']

split_datas_into_cultivators(raw_dataset)


#visualize_datas(, 'Dataset Visualisation')
