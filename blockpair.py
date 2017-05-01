import numpy as np
from pydtk.tree import Tree
from pydtk.dtk import DT
from pydtk.operation import fast_shuffled_convolution
import time as time

'''
Class to rapresent a pair of blocks. Every block contains a group of trees (TreesA and TreesB) 

'''
class BlockPair:	
	def __init__(self, dim = 4096):
		self.dim = dim
		self.treesA = []
		self.treesB = []

	'''
	method to transform a syntatix tree (made of words) in a dt
	'''
	def fromTree2DT(self,X,dt):
		tr = Tree(string=X)
		return dt.dt(tr)

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
	def blockOneToDT(self, dt):
		T = []
		if(len(self.treesA) > 0):
			for t in range(len(self.treesA)):
				T.append(self.fromTree2DT(self.treesA[t],dt))	
			for i in range(len(T)):
				if (i != 0):
					total = mySum(total, T[i])
				else:
					total = T[i]
			return total/np.linalg.norm(total)
			

	'''
	method to transform an entire block (block B) of tree in a DT after the transformation of evert tree in a dt
	'''
	def blockTwoToDT(self, dt):
		T = []
		if(len(self.treesB) > 0):
			for t in range(len(self.treesB)):
				T.append(self.fromTree2DT(self.treesB[t],dt))
			for i in range(len(T)):
				if (i != 0):
					total = mySum(total, T[i])
				else:
					total = T[i]
			return total/np.linalg.norm(total)
	
	'''
	Method to concatenate the 2 blocks (transformed in DT) (but before they do the transormation of the tree in dt)
	'''
	def Concatenation(self, dt):
		return np.concatenate([self.blockOneToDT(dt),self.blockTwoToDT(dt)],axis=0)

'''
Method to call the concatenation method of bp (made for PreEmbedder)
'''

def Concatenation(bp, dt):
	return bp.Concatenation(dt)

def mySum(list1, list2):
	listSum = []
	if len(list1) != len(list2):
		return None;
	else:
		for i in range(len(list1)):
			one = np.float32(list1[i])
			two = np.float32(list2[i])
			listSum.append(sum([one,two]))
		return listSum



