from convolutions import *
import numpy as np
import numpy.random as rnd
from keras import backend as K
from vectors import *
from trees import *
from datetime import datetime


class DT:
	#seed random gen, dim tensor dimensions, lambd decay,lexicalized consider leaves, mu mean, va variance
	def __init__(self,seed = 0,dim = 1024,lambd=1,lexicalized=False,mu = 0,va = 1,operator = circular_convolution):
		self.dim = dim
		self.lambd = lambd
		self.vector_generator=Vector_generator(seed,dim=dim)
		self.lexicalized = lexicalized
		self.operator = operator

		self.permutations=Vector_generator.permutations(dim=dim,seed=seed)

	#compute dt, t is nltk.tree.Tree
	def dt(self,t, to_penn=True):
		tensor = K.zeros((self.dim,))
		if to_penn:
				tree = Tree.from_penn(t)
		else:
				tree = t
		self.__recursion(tree,tensor)
		return tensor.eval()
	#compute S(n) with dfs
	def __recursion(self,t,s):
		#print t
		res = K.zeros((self.dim,))

		if len(t)>0:
			preterminal = True
			for i in range(len(t)):
				child_v = self.vector_generator.get_random_vector(t[i].label)
				if len(t)>0:
					child_v = self.vector_generator.get_random_vector(t[i].label) + (self.lambd * self.__recursion(t[i], s))
					preterminal = False

				res = child_v if (i == 0) else self.operator(res, child_v,self.permutations)
			if (not preterminal) or self.lexicalized:
				#print t
				res = self.operator(self.vector_generator.get_random_vector(t.label), res,self.permutations)

				if K.backend() == 'theano':
					s.set_value(np.float32((s+res).eval()))
				else:
					#print 'tf'
					s.assign((s+res).eval()).eval()
			else:
				res = K.zeros((self.dim,))
		return res
