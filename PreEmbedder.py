import numpy as np
import random 
from blockpair import *


'''
PreEmbedder take as inputs the Y of a net, the features, a function and the X of a net.
It applies the function to the X to transofrm it in something else (like a DT for example)
and then divide the X and the Y in blocks of K elements
It return a list of K-lenght blocks and a list of K-lenght Y after it shuffled the order of the blocks 
(but it mantain the corrispondece of blocks and Ys)
'''

cache = {}


def PreEmbedder(Yargs, features, function, Xargs):
	k=50
	count = 0
	y = 0
	DT = []
	blocksDT = []
	k_lenght_blocksDT = []
	k_lenght_Yargs=[]
	k_lenght_Ylist = []
	K_lenght_blocklist = []
	
	for i in range(len(Xargs)):
		DT = cachesBlocksFromBlockPair(function)(Xargs[i])
		blocksDT.append(np.concatenate([features[i], DT], axis = 0))
	for block in blocksDT:
		K_lenght_blocklist.append(block)
		k_lenght_Ylist.append(Yargs[y])
		count = count + 1
		if(count == k):
			k_lenght_blocksDT.append(K_lenght_blocklist)
			k_lenght_Yargs.append(k_lenght_Ylist)
			count = 0
			K_lenght_blocklist = []
			k_lenght_Ylist = []
		y = y + 1
	shufflelist = list(zip(k_lenght_blocksDT, k_lenght_Yargs))
	random.shuffle(shufflelist)
	k_lenght_blocksDT, k_lenght_Yargs = zip(*shufflelist)
		
	return k_lenght_blocksDT, k_lenght_Yargs	


#funzione per decorare la funzione che passo al preembedder che serve a cachare i risultati che ho ottenuto
def cachesBlocksFromBlockPair(func):
	def _func(*args):
		x = args[-1]
		#lavorare il block per ottenere la chiave della cache assumiamo per ora che sia str(treesBlock)
		valA = cache.get(str(x.treesA), None)  #restituisce il valore del dizionario con chiave primo argomento oppure il secondo argomento (default)
		valB = cache.get(str(x.treesB), None)
		if valA != None and valB != None:
			return np.concatenate([valA, valB])
		elif valA == None and valB != None:
			valA = x.blockOneToDT()
			cache[str(x.treesA)] = valA
			return np.concatenate([valA, valB])
		elif valA != None and valB == None:
			valB = x.blockTwoToDT()
			cache[str(x.treesB)] = valB
			return np.concatenate([valA, valB])
		else: #valA == None and valB == None
			value = func(x)
			cache[str(x.treesA)] = value[:4096]
			cache[str(x.treesB)] = value[4096:]
			return value
	return _func
	

'''

Passo successivo: Sto forzando il PreEmbedder ad avere come input solo una lista di BlockPair, posso definire un dizionario dove i tipi sono le chiavi e le funzioni decoratrici sono i valori così da selezionare il decoratore giusto a seconda del tipo

'''		
		
		










