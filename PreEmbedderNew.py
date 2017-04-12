


class PreEmbedderNew:

    def __init__(self,  function, Xs_with_trees, Xs_additiona_features = None,  Ys, batchDim = 50):
       self._Ys = Ys
       self._Xs_with_trees = Xs_with_trees
       self._Xs_additiona_features = Xs_additiona_features
       self._function = function
       self._batchDim = batchDim
       self._batchIndex = 0


    def hasNextBatch(self):
        return len(self._Xargs) > self._batchIndex * self._batchDim

    def nextBacth(self):
        batchX = []
        batchY = []

        # Fill and embed batchX and Y

        self._batchIndex = self._batchIndex + 1
        return batchX,batchY

    def reset(self):
        #reshaffle examples

        #restart indexing
        self._batchIndex = 0
