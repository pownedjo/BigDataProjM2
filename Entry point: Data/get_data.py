import csv
import urllib.request

'''
Fetching data from internet
'''
url = 'http://lipn.univ-paris13.fr/~grozavu/BgDOuv/dataProject/8%20-%20wine/wine.data'

csv_cont = urllib.request.urlopen(url)
csv_cont = csv_cont.read()	# .decode('utf-8') if we want to work with strings

'''
Saving data locally
'''
with open('./wine.data', 'wb') as out:
    out.write(csv_cont)


