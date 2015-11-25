from sklearn import datasets
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt


results = []

## Cultivateurs lists
cultivateur1 = []
cultivateur2 = []
cultivateur3 = []

## Wine Components lists
composant1 = []
composant2 = []
composant3 = []
composant4 = []
composant5 = []
composant6 = []
composant7 = []
composant8 = []
composant9 = []
composant10 = []
composant11 = []
composant12 = []
composant13 = []


def parse_file(file):
	with open(file) as inputfile:
		for line in inputfile:
			results.append(line.strip().split(','))
	
	
## GET A LIST FOR EACH CULTIVATEURS
def parse_into_cultivars(results):
	list_test = []
	for list_test in results:
		if list_test[0] == '1':
			cultivateur1.append(list_test)
		elif list_test[0] == '2':
			cultivateur2.append(list_test)
		else:
			cultivateur3.append(list_test)
			

## GET A LIST FOR EACH WINE COMPOSANTS
def parse_into_wine_composants(results):
	single_list = []
	for single_list in results:
		composant1.append(single_list[1])	## Index 0 = Cultivateur Value
		composant2.append(single_list[2])
		composant3.append(single_list[3])
		composant4.append(single_list[4])
		composant5.append(single_list[5])
		composant6.append(single_list[6])
		composant7.append(single_list[7])
		composant8.append(single_list[8])
		composant9.append(single_list[9])
		composant10.append(single_list[10])
		composant11.append(single_list[11])
		composant12.append(single_list[12])
		composant13.append(single_list[13])


def arrange_datas_composants_by_cultivars(cultivar):
	single_list = []
	datas_test = [[]]
	#x = 1
	for single_list in cultivar:
		for x in range(1, 13):
			datas_test.append(single_list[x])
	print datas_test
	
			
## Datas Visualisation - Display a List of Lists
def visualize_datas(dataset):
	fig = plt.figure()
	fig.suptitle('Datas Visualizations with Matplotlib', fontsize=18)
	plt.xlabel('Abscisse', fontsize=14)
	plt.ylabel('Ordonee', fontsize=14)
	
	for list_test in dataset:
		plt.plot(list_test)
	

	plt.show()


## Feed with ONLY cultivateur lists
def arranging_datas_cultivars(cultivateurX):	
	for list_test in cultivateurX:
		list_test.pop(0)	## Remove first index (cultivateur value)


def main():
	print '###############################################'
	print 'Launching big Data project with python'
	print 'Authors :'
	print 'Jordan TETE'
	print 'Thomas SOUVANNASAO'
	print '###############################################\n'

	parse_file('wine.txt')
	parse_into_cultivars(results)
	
	arrange_datas_composants_by_cultivars(cultivateur1)
	
	#parse_into_wine_composants(results)
	parse_into_wine_composants(cultivateur1)
	
	dataset = []
	dataset.append(composant1)
	dataset.append(composant2)
	dataset.append(composant3)
	dataset.append(composant4)

	
	visualize_datas(dataset)
	
	



#########################
### START MAIN SCRIPT ###
#########################
main()