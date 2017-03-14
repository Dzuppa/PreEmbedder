import numpy as np
import random 
from blockpair import *


'''
PreEmbedder take as inputs the Y of a net, the features, a function and the X of a net.
It applies the function to the X to transofrm it in something else (like a DT for example)
and then divide the X and the Y in blocks of K elements
It return a list of K-lenght blocks and a list of K-lenght Y after it shuffled the order of the blocks 
(but it mantain the corrispondece of blocks and Ys)
'''


def PreEmbedder(turn, Yargs, features, function, Xargs):
	k=5
	count = 0
	y = 0
	blocksDT = []
	k_lenght_blocksDT = []
	k_lenght_Yargs=[]
	k_lenght_Ylist = []
	K_lenght_blocklist = []
	for i in range(len(Xargs)):
		blocksDT.append(np.concatenate([features[i],function(Xargs[i],turn)], axis = 0))
		if(len(blocksDT)) == 10:
			break
	for block in blocksDT:
		K_lenght_blocklist.append(block)
		k_lenght_Ylist.append(Yargs[y])
		count = count + 1
		if(count == k):
			k_lenght_blocksDT.append(K_lenght_blocklist)
			k_lenght_Yargs.append(k_lenght_Ylist)
			count = 0
			K_lenght_blocklist = []
			k_lenght_Ylist = []
		y = y + 1
	shufflelist = list(zip(k_lenght_blocksDT, k_lenght_Yargs))
	random.shuffle(shufflelist)
	k_lenght_blocksDT, k_lenght_Yargs = zip(*shufflelist)
		
	return k_lenght_blocksDT, k_lenght_Yargs	

#main






