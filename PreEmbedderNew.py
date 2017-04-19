
import random
from keras_dt import *
import numpy as np

cache = {}

class PreEmbedderNew:

    def __init__(self,  function, Xs_with_trees, Ys, Xs_additional_features = None, batchDim = 50):
        self._Ys = Ys
        self._Xs_with_trees = Xs_with_trees
        self._Xs_additional_features = Xs_additional_features
        self._function = function
        self._batchDim = batchDim
        self._batchIndex = 0
        self._listOfIndexes = list(range(len(Xs_with_trees)))
        self.dt = DT(dim=32, lexicalized=True)
        random.shuffle(self._listOfIndexes)



    def hasNextBatch(self):
        return len(self._Xs_with_trees) > self._batchIndex * self._batchDim

    def nextBacth(self):
        batchX = []
        batchY = []
        max = (self._batchIndex + 1) * (self._batchDim) if (self._batchIndex + 1) * (self._batchDim) < len(self._listOfIndexes) else len(self._listOfIndexes) - 1
        print("max: ", max)
        print("list 4 for: ", self._listOfIndexes[self._batchIndex * self._batchDim:max])
        for i in self._listOfIndexes[self._batchIndex * self._batchDim:max] :
            #transform X_i
            batchX.append(np.concatenate([cachesBlocksFromBlockPair(self._function)(self.dt,self._Xs_with_trees[i]), self._Xs_additional_features[i]], axis=0))
            batchY.append(self._Ys[i])
        self._batchIndex = self._batchIndex + 1
        return batchX,batchY

    def reset(self):
        #reshaffle examples
        random.shuffle(self._listOfIndexes)
        #restart indexing
        self._batchIndex = 0



#funzione per decorare la funzione che passo al preembedder che serve a cachare i risultati che ho ottenuto
def cachesBlocksFromBlockPair(func):
	def _func(*args):
		x = args[-1]
		dt = args[-2]
		#lavorare il block per ottenere la chiave della cache assumiamo per ora che sia str(treesBlock)
		valA = cache.get(str(x.treesA), None)  #restituisce il valore del dizionario con chiave primo argomento oppure il secondo argomento (default)
		valB = cache.get(str(x.treesB), None)
		if valA is not None and valB is not None:
			print("..")
			return np.concatenate([valA, valB])
		elif valA is None and valB is not None:
			valA = x.blockOneToDT(dt)
			print(".")
			cache[str(x.treesA)] = valA
			return np.concatenate([valA, valB])
		elif valA is not None and valB is None:
			valB = x.blockTwoToDT(dt)
			print(".")
			cache[str(x.treesB)] = valB
			return np.concatenate([valA, valB])
		else: #valA is None and valB is None
			value = func(x,dt)
			print("calculated")
			cache[str(x.treesA)] = value[:x.dim]
			cache[str(x.treesB)] = value[x.dim:]
			return value
	return _func
