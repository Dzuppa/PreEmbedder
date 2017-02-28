#from keras import backend as K
import numpy as np
from vectors import *
from scipy.linalg import circulant
# according this review http://scottsievert.com/blog/2016/07/01/numpy-gpu/
# numpy fast computes FFT without acceleration (theano use numpy for fft)

#circular convolution, x,y arrays, permutations is not used
def circular_convolution(x,y,permutations=None):
	if type(x) != np.ndarray:
		x=x.eval()
	if type(y) != np.ndarray:
		y=y.eval()
	return np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(y)))
#circular convolution, x,y arrays, permutations is used
def shuffled_circular_convolution(x,y,permutations=None):
	if type(x) != np.ndarray:
		x=x.eval()
	if type(y) != np.ndarray:
		y=y.eval()
	#print permutations[0],'\n',permutations[1]
	return circular_convolution(x,y[permutations[1]],permutations)

def permutation_matrices(N):
	I = np.eye(N)
	p1,p2 = Vector_generator.permutations(N) #using that 99,5% cosine
	#p1 = circulant(I[:,1]) #99,28%
	#p2 = p1.T
	#print p1,'\n',p2
	return I[p1], I[p2]

def cc_circulant(x,y):
	#permuation matrices
	Phi1,Phi2 = permutation_matrices(x.shape[0])
	A = circulant(x) #circulant matrix
	B = y
	
	#print Phi1,'\n',Phi2
	#shuffled circular convolution
	return A.dot(Phi2).dot(B)
if __name__ == '__main__':
	x=np.array([1,2,3,5])
	y=np.array([2,3,4,7])
	cc1,cc2= cc_circulant(x,y),shuffled_circular_convolution(x,y,Vector_generator.permutations(dim=4))
	print(cc1.dot(cc2)/(np.linalg.norm(cc1,2)*np.linalg.norm(cc2,2)))
