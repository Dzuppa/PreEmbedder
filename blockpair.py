import numpy as np
from keras_dt import *
from trees import *
import time as time

'''
Class to rapresent a pair of blocks. Every block contains a group of trees (TreesA and TreesB) 

'''
class BlockPair:	
	def __init__(self):
		self.treesA = []
		self.treesB = []

	'''
	method to transform a syntatix tree (made of words) in a dt
	'''
	def fromTree2DT(self,X):
		dt = DT(dim=4096, lexicalized=True)
		return dt.dt(X,to_penn = True)

	def addTreeToBlockOne(self,tree):
		self.treesA.append(tree)

	def addTreeToBlockTwo(self,tree):
		self.treesB.append(tree)

	def clearTreesFromBlockOne(self):
		self.treesA = []

	def clearTreesFromBlockTwo(self):
		self.treesB = []
	
	'''
	method to transform an entire block (block A) of tree in a DT after the transformation of evert tree in a dt
	'''
	def blockOneToDT(self):
		T = []
		if(len(self.treesA) > 1):
			for t in range(len(self.treesA)):
				T.append(self.fromTree2DT(self.treesA[t]))	
			for i in range(len(T)):
				if (i != 0):
					total = np.sum([total, T[i]],axis=0)
				else:
					total = T[i]
			return total/np.linalg.norm(total)
			

	'''
	method to transform an entire block (block B) of tree in a DT after the transformation of evert tree in a dt
	'''
	def blockTwoToDT(self):
		T = []
		if(len(self.treesB) > 1):
			for t in range(len(self.treesB)):
				T.append(self.fromTree2DT(self.treesB[t]))
			for i in range(len(T)):
				if (i != 0):
					total = np.sum([total, T[i]],axis=0)
				else:
					total = T[i]
			return total/np.linalg.norm(total)
	
	'''
	Method to concatenate the 2 blocks (transformed in DT) (but before they do the transormation of the tree in dt)
	'''
	def Concatenation(self):
		return np.concatenate([self.blockOneToDT(),self.blockTwoToDT()],axis=0)

'''
Method to call the concatenation method of bp (made for PreEmbedder)
'''

def Concatenation(bp):
	return bp.Concatenation()


