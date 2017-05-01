
import random
from pydtk.tree import Tree
from pydtk.dtk import DT
from pydtk.operation import fast_shuffled_convolution
import numpy as np
import cProfile


cache = {}

class PreEmbedderNew:

    def __init__(self,  function, Xs_with_trees, Ys, Xs_additional_features = None, batchDim = 50, Y_test_table=None):
        self._Ys = Ys
        self._Xs_with_trees = Xs_with_trees
        self._Xs_additional_features = Xs_additional_features
        self._function = function
        self._batchDim = batchDim
        self._Y_test_table = Y_test_table
        self._batchIndex = 0
        self._listOfIndexes = list(range(len(Xs_with_trees)))
        self.dt = DT(dimension=4096, LAMBDA= 0.6, operation=fast_shuffled_convolution)
        random.shuffle(self._listOfIndexes)



    def hasNextBatch(self):
        return len(self._Xs_with_trees) > self._batchIndex * self._batchDim

    def nextBacth(self):
        batchX = []
        batchY = []
        features = []
        Test_tables = []
        max = (self._batchIndex + 1) * (self._batchDim) if (self._batchIndex + 1) * (self._batchDim) < len(self._listOfIndexes) else len(self._listOfIndexes) - 1
        for i in self._listOfIndexes[self._batchIndex * self._batchDim:max] :
            #transform X_i
            batchX.append(cachesBlocksFromBlockPair(self._function)(self.dt,self._Xs_with_trees[i]))
            batchY.append(self._Ys[i])
            features.append(self._Xs_additional_features[i])
            if self._Y_test_table is not None:
                Test_tables.append(self._Y_test_table[i])
        self._batchIndex = self._batchIndex + 1
        return batchX,batchY, Test_tables, features

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
			return np.concatenate([valA, valB])
		elif valA is None and valB is not None:
			valA = x.blockOneToDT(dt)
			cache[str(x.treesA)] = valA
			return np.concatenate([valA, valB])
		elif valA is not None and valB is None:
			valB = x.blockTwoToDT(dt)
			cache[str(x.treesB)] = valB
			return np.concatenate([valA, valB])
		else: #valA is None and valB is None
			value = func(x,dt)
			cache[str(x.treesA)] = value[:x.dim]
			cache[str(x.treesB)] = value[x.dim:]
			return value
	return _func
