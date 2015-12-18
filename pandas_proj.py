import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


columns = ['Cultivar', 'Alcohol', 'Malic Acid', 'Ash', 'Alkalinity of Ash', 'Magnesium', 'Total phenol', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

raw_dataset = pd.read_csv(open('wine.txt'), names=columns)
description_of_dataset = raw_dataset.describe()
print description_of_dataset



## Datas Visualisation - Display a List of Lists
def visualize_datas(dataset):
    fig = plt.figure()
    fig.suptitle('Datas Visualizations with Matplotlib', fontsize=18)
    plt.xlabel('Abscisse', fontsize=14)
    plt.ylabel('Ordonee', fontsize=14)

    for list_test in dataset:
        plt.plot(list_test, 'ro') # Nuage de points

    plt.show()  ## show graph


## MAIN