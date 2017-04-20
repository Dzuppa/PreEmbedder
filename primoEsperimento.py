from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.optimizers as opt
import numpy as np
import sys
import corpus_reader as cr
import PreEmbedder as pe
import blockpair as bp
from keras_dt import *

max_num_of_words = 40000
inputs = []
base = "../standard_split_reduced_3/"

X_train,X_train_specific_features,Y_train,max_value,_,Pairs = cr.read_corpus(base+"Dev-training.txt", base+"blocks.txt", 3,n_of_words=max_num_of_words)
'''
X_test,X_test_specific_features,Y_test,max1,Y_test_table,PairsTest = cr.read_corpus(base+"Dev-testing_full_grouped.txt", base+"blocks.txt", 3,max_lenght_of=max_value,n_of_words=max_num_of_words)
'''
model = Sequential()

model.add(Dense(input_dim=73,output_dim=3, activation='sigmoid'))
dt = DT(dim=32, lexicalized=True)

optim = opt.adam(0.0004, 0.9, 0.999)
print("compiling")
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
print("done")
for i in range(1,500):
	versus = 0
	changed = 0
	epsilon = 0.00
	sys.stdout.flush()
	print("i:", i)
	inputs, targets = pe.PreEmbedder(dt,Y_train,X_train_specific_features, bp.Concatenation, Pairs)
	print("Embedding done")
	#inputs_test, targets_test = pe.PreEmbedder(Y_test,X_test_specific_features,bp.Concatenation, PairsTest)
	
	inputs = np.asarray(inputs)
	targets = np.asarray(targets)
	
	#inputs_test = np.asarray(inputs_test)
	#targets_test = np.asarray(targets_test)
	
	for t in range(len(inputs)):
		print("fitting")
		model.fit(inputs[t], targets[t], nb_epoch=1, batch_size=50,class_weight=[0.999,0.001],verbose=1)
		print("prediction")
		system_predictions = final_model.predict(inputs_test[i],targets_test[i], batch_size=300, verbose=1)
		'''
		for j in np.random.random_integers(0,len(system_predictions),10):
			print(system_predictions[j])
		print("Generating tables ", i)
		oracle_table = e.generate_table(Y_test_table, Y_test)    
		system_table = e.generate_table(Y_test_table, system_predictions)    
		print("Computing performance ", i)
		recall_at_k_res = e.recall_at_k(oracle_table, system_table, k_s = [1,2,5,10,100,1000])
		
		if(changed != 0):
			versus = 0
			for k in range(3):
				if(recall_at_k_res[i] < last_kres[i] or (recall_at_k_res[i] - last_kres[i]) < epsilon):
					versus = versus + 1
				if(versus > 2):
					K.set_value(optim.lr, 0.5 * K.get_value(optim.lr))
					changed = 0
		else:
			changed = 1;
	
		for w in range(len(recall_at_k_res)):
			last_kres[w] = recall_at_k_res[w]	
  		'''
    
