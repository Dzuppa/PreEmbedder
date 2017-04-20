import numpy as np
import random 
from blockpair import *
from datetime import datetime


'''
PreEmbedder take as inputs the Y of a net, the features, a function and the X of a net.
It applies the function to the X to transofrm it in something else (like a DT for example)
and then divide the X and the Y in blocks of K elements
It return a list of K-lenght blocks and a list of K-lenght Y after it shuffled the order of the blocks 
(but it mantain the corrispondece of blocks and Ys)
'''

cache = {}


def PreEmbedder(dt, Yargs, features, function, Xargs):
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
		print("creo: ",i)
		DT = cachesBlocksFromBlockPair(function)(dt, Xargs[i])
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
		dt = args[-2]
		#lavorare il block per ottenere la chiave della cache assumiamo per ora che sia str(treesBlock)
		valA = cache.get(str(x.treesA), None)  #restituisce il valore del dizionario con chiave primo argomento oppure il secondo argomento (default)
		valB = cache.get(str(x.treesB), None)
		if valA is not None and valB is not None:
			print("..", datetime.now().time())	
			return np.concatenate([valA, valB])
		elif valA is None and valB is not None:
			valA = x.blockOneToDT(dt)
			print(".", datetime.now().time())
			cache[str(x.treesA)] = valA
			return np.concatenate([valA, valB])
		elif valA is not None and valB is None:
			valB = x.blockTwoToDT(dt)
			print(".", datetime.now().time())
			cache[str(x.treesB)] = valB
			return np.concatenate([valA, valB])
		else: #valA is None and valB is None
			value = func(x,dt)
			cache[str(x.treesA)] = value[:x.dim]
			cache[str(x.treesB)] = value[x.dim:]
			print("created ",datetime.now().time())
			return value
	return _func
	

'''

Passo successivo: Sto forzando il PreEmbedder ad avere come input solo una lista di BlockPair, posso definire un dizionario dove i tipi sono le chiavi e le funzioni decoratrici sono i valori così da selezionare il decoratore giusto a seconda del tipo

'''		



def randomize3ListTogheter(list1, list2, list3):
	'''
	randomize 3 lists of the same size manteining the ordering 
	'''
	pos2 = 0
	for i in range(400000):
		random.seed(datetime.now())
		pos1 = random.randrange(0,len(list1),1)
		while(pos2 == pos1):
			pos2 = random.randrange(0,len(list1),1)
		list1 = changepos(pos1,pos2,list1)
		list2 = changepos(pos1,pos2,list2)
		list3 = changepos(pos1,pos2,list3)
	return list1, list2, list3



def changepos(pos1, pos2, lis):
	t = lis[pos1]
	lis[pos1] = lis[pos2]
	lis[pos2] = t
	return lis
		






