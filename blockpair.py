import numpy as np
from keras_dt import *
from trees import *
import time as time

'''

Class to rapresent a pair of blocks. Every block contains a group of trees (TreesA and TreesB) 
and a dictionary with the calculated DT of every tree

'''
class BlockPair:	
	def __init__(self):
		self.treesA = []
		self.treesB = []
		self.SavedDT = {}

	'''
	method to transform a syntatix tree (made of words) in a dt
	'''
	def fromTree2DT(self,X):
		dt = DT(dim=32, lexicalized=True)
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
	def blockOneToDTWithTransformation(self):
		T = []
		if(len(self.treesA) > 1):
			for t in range(len(self.treesA)):
				if(self.treesA[t] not in self.SavedDT):
					self.SavedDT[self.treesA[t]] = self.fromTree2DT(self.treesA[t])
				T.append(self.SavedDT[self.treesA[t]])	
			self.clearTreesFromBlockOne()
			for a in T:
				self.addTreeToBlockOne(a)
			for i in range(len(self.treesA)):
				if (i != 0):
					total = np.sum([total, self.treesA[i]],axis=0)
				else:
					total = self.treesA[i]
			return total/np.linalg.norm(total)

	'''
	method to transform an entire block (block B) of tree in a DT after the transformation of evert tree in a dt
	'''
	def blockTwoToDTWithTransformation(self):
		T = []
		if(len(self.treesB) > 1):
			for t in range(len(self.treesB)):
				if(self.treesB[t] not in self.SavedDT):
					self.SavedDT[self.treesB[t]] = self.fromTree2DT(self.treesB[t])
				T.append(self.SavedDT[self.treesB[t]])	
			self.clearTreesFromBlockTwo()
			for a in T:
				self.addTreeToBlockTwo(a)
			for i in range(len(self.treesB)):
				if (i != 0):
					total = np.sum([total, self.treesB[i]],axis=0)
				else:
					total = self.treesB[i]
			return total/np.linalg.norm(total)
	
	def blockOneToDT(self):
		if(len(self.treesA) > 1):
			for i in range(len(self.treesA)):
				if (i != 0):
					total = np.sum([total, self.treesA[i]],axis=0)
				else:
					total = self.treesA[i]
			return total/np.linalg.norm(total)

	def blockTwoToDT(self):
		if(len(self.treesB) > 1):
			for i in range(len(self.treesB)):
				if (i != 0):
					total = np.sum([total, self.treesB[i]],axis=0)
				else:
					total = self.treesB[i]
				return total/np.linalg.norm(total)
	'''
	Method to concatenate the 2 blocks (transformed in DT) (but before they do the transormation of the tree in dt)
	'''
	def ConcatenationWithTransformation(self):
		return np.concatenate([self.blockOneToDTWithTransformation(),self.blockTwoToDTWithTransformation()],axis=0)

	def Concatenation(self):
		return np.concatenate([self.blockOneToDT(),self.blockTwoToDT()],axis=0)
'''
Method to call the concatenation method of bp (made for PreEmbedder)
'''

def Concatenation(bp, i):
	if(i == 0):
		return bp.ConcatenationWithTransformation()
	else:
		return bp.Concatenation()


