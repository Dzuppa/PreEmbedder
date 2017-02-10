import numpy as np
import random 

'''
Uso come dati di input di prova degli alberi qualsiasi
'''

Trees = {} #dizionario degli alberi che conterrà coppie chiave (gruppo di alberi) e valore (codifica)
tree = [] #lista che uso per leggere gli alberi dal file
T = [] #lista di appoggio per fare i gruppi di alberi
k = 3 #grandezza dei gruppi di alberi (statica in questo caso)


#lettura degli alberi da file
with open('SampleInput.dat', 'r') as f:
	tree = [line.replace('\n', '') for line in f.readlines()]

if('' in tree):
	tree.remove('')

#conto quanti alberi ho preso dal file
n = len(tree)
a = 0;  #variabile intera di appoggio per formare i gruppi di alberi

#credo il dizionario mettendo come chiave il gruppo di alberi e come valore la loro codifica (in questo caso semplicemente un array di indici)
for i in tree:
	T.append(i)
	a = a+1
	if(a == k): #or i == tree[-1]):
		Trees[str(T)] = np.unique(T,return_inverse=True)[1]
		T = []
		a = 0


#funzione che prende in input un gruppo di alberi e restituisce il dot product tra loro
def functionDot(X):
	result = 1
	for item in X:
		result = np.dot(result, item)
	return result

#PreEmbedder che prende in input Y, la funzione con cui creare i dt e il gruppo di alberi (per creare i dt faccio il dot product della codifica degli alberi)	
def PreEmbedder(Yargs, function, *Xargs):
	result = function(*Xargs)
	return result, Yargs	

#loop che randomizza l'accesso al dizionario ed avvia il PreEmbedder (da ripetere)
for i in range(10):
	keys = list(Trees.keys())
	random.shuffle(keys)
	print(keys)
	
	for key in keys:
		v = Trees[key]
		print (PreEmbedder(1, functionDot, v))
		#aggiungere una variabile a cui assegnare il valore del return del PreEmbedder ed il fit
	
	
#da inserire la rete