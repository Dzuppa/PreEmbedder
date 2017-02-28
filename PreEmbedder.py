import numpy as np
#import corpus_reader as cr
import random 
from keras_dt import *
from trees import *

'''
REMINDER
per fare l'hash delle liste è necessario prima trasformarle in stringhe
facendo quindi hash(str(list)) 
'''

k = 3
ToSave = {}


#Problemi: riuscire a convertire il dt da stringa a vettore quando lo si legge da file

def SaveData(dictionary):
	with open('SavedTree.txt','w')as data:
		for key in dictionary.keys():
			data.write(key + ":"+ str(dictionary[key]).replace('\n', ' ') + "\n")	


def readDataFile():
	with open('SavedTree.txt','r')as data:
		dictionary = {}
		for line in data:
			dictionary[line.split(':')[0]] = line.split(':')[1].strip()
		print("readed:",dictionary)
		return dictionary

def functionDot(X):
	result = 1
	for item in X:
		result = np.dot(result, item)
	return result
		
def fromTree2DT(X):
	dt = DT(dim=32, lexicalized=True)
	print(dt)
	print(dt.dt(X,to_penn = True))
	return dt.dt(X,to_penn = True)


def PreEmbedder(Yargs, function, *Xargs):
	Trees = []
	sums = []
	T = []
	blocks = {}
	a = 0
	for i in Xargs[0]:
		#print ("i: ",i)
		if not i in ToSave:
			result = function(i)
			Trees.append(result)
			ToSave[i] = result
			#print("Saved:",i)
		else:
			Trees.append(ToSave[i])
			#print("taked from the cache: ",i)
	for i in Trees:
		T.append(i)
		a = a+1
		if(a == k): 
			blocks[hash(str(T))] = np.sum([T[0],T[1],T[2]], axis = 0) / np.linalg.norm(np.sum([T[0],T[1],T[2]], axis = 0))
			T = []
			a = 0
	print(blocks)
	return blocks, Yargs	

#main

with open('SampleInput.dat', 'r') as f:
	tree = [line.replace('\n', '') for line in f.readlines()]

if('' in tree):
	tree.remove('')

'''
import sys
import mainscript

part1Cache = None
if __name__ == "__main__":
    while True:
        if not part1Cache:
            part1Cache = mainscript.part1()
        mainscript.part2(part1Cache)
        print "Press enter to re-run the script, CTRL-C to exit"
        sys.stdin.readline()
        reload(mainscript)
'''

#i != 0 da eliminare, serve solo per non far caricare il file vuoto ed andare in errore alla prima esecuzione

for i in range(2):
	if not ToSave and i != 0:
		ToSave = readDataFile()
		print("ToSave:",ToSave)
	else:
		PreEmbedder([],fromTree2DT,tree)
		

SaveData(ToSave)





'''
for i in range(10):
	keys = list(Trees.keys())S
	random.shuffle(keys)
	print(keys)
	
	for key in keys:
		v = Trees[key]
		print (PreEmbedder(1, functionDot, v))
	
'''
