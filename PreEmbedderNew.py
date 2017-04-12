
import random

class PreEmbedderNew:

    def __init__(self,  function, Xs_with_trees, Xs_additional_features = None,  Ys, batchDim = 50):
        self._Ys = Ys
        self._Xs_with_trees = Xs_with_trees
        self._Xs_additional_features = Xs_additional_features
        self._function = function
        self._batchDim = batchDim
        self._batchIndex = 0
        self._listOfIndexes = range(len(Xs_with_trees))
        random.shuffle(self._listOfIndexes)



    def hasNextBatch(self):
        return len(self._Xargs) > self._batchIndex * self._batchDim

    def nextBacth(self):
        batchX = []
        batchY = []
        max = self._batchIndex * (self._batchDim + 1) if self._batchIndex * (self._batchDim + 1) < len(self._listOfIndexes) else len(self._listOfIndexes) - 1

        for i in self._listOfIndexes[self._batchIndex * self._batchDim:max] :
            #transform X_i
            batchX.append(self._function.embed(self._Xs_with_trees[i]) + self._Xs_additional_features[i])
            batchY.append(self._Ys[i])
        self._batchIndex = self._batchIndex + 1
        return batchX,batchY

    def reset(self):
        #reshaffle examples
        random.shuffle(self._listOfIndexes)
        #restart indexing
        self._batchIndex = 0
