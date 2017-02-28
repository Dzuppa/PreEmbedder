#from keras import backend as K
import numpy as np
import sys
'''
random vector generator(versors)
'''
class Vector_generator:
	def __init__(self,seed=0,dim=1024,mu=0,std=1):
		self.seed=seed
		self.dim=dim
		self.mu=np.zeros(self.dim)
		self.std=np.eye(self.dim)*(1/float(self.dim))
		self.cache={}

	def get_random_vector(self,label):
		if label in self.cache:
			#print label
			return self.cache[label]
		#if seed is hash adjust it (can be negative)
		#seed = hash(label)
		from hashlib import sha256
		#data = np.random.rand(1000)
		hashl = sha256(label.encode('utf-8'))
		seed = np.frombuffer(hashl.digest(), dtype='uint32')
		#seed=np.frombuffer(seed, dtype='uint32')
			#print seed
	  	#self.seed=0

		np.random.seed(seed)
		
		vect = np.random.multivariate_normal(self.mu,self.std)
		#print vect

		self.cache[label]=vect
		return vect
	@staticmethod
	def permutation(dim=1024,seed=0,permutation=None):

		np.random.seed(seed)
		perm=np.random.permutation(dim)
		while(not Vector_generator.test_permutation(perm,permutation)):
			perm=np.random.permutation(dim)
		return perm
	@staticmethod
	def test_permutation(perm1,perm2=None):
		basic=np.arange(perm1.shape[0])
		shuffled=basic.copy()

		for i in range(basic.shape[0]):
			#print i
			if (shuffled[perm1] == basic).all():
				#print 'f'
				return False
			if not perm2 is None:
				if (shuffled == basic[perm2]).all():
					return False
				if (basic == shuffled[perm2]).all():
					return False
		return True
	@staticmethod
	def permutations(dim=1024,seed=0):
		perm1 = Vector_generator.permutation(dim=dim,seed=0)
		#in accordo con l'implementazione java (quella python di ferrone utilizza 123)
		perm2 = Vector_generator.permutation(dim=dim,seed=1,permutation=perm1)
		return (perm1,perm2)
