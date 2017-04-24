from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.optimizers as opt
import numpy as np
import sys
import corpus_reader as cr
import PreEmbedderNew as pe
import blockpair as bp
import cProfile
import evaluator as e
import keras.backend as K


max_num_of_words = 40000
inputs = []
base = "../../standard_split_reduced_3/"

X_train,X_train_specific_features,Y_train,max_value,_,Pairs = cr.read_corpus(base+"Dev-training.txt", base+"blocks.txt", 3,n_of_words=max_num_of_words)

X_test,X_test_specific_features,Y_test,max1,Y_test_tables,PairsTest = cr.read_corpus(base+"Dev-testing_full_grouped.txt", base+"blocks.txt", 3,max_lenght_of=max_value,n_of_words=max_num_of_words)



model = Sequential()

model.add(Dense(input_dim=73,output_dim=2, activation='sigmoid'))

penew = pe.PreEmbedderNew(bp.Concatenation, Pairs, Y_train, X_train_specific_features)

peTest = pe.PreEmbedderNew(bp.Concatenation, PairsTest, Y_test, X_test_specific_features, Y_test_table = Y_test_tables)

optim = opt.adam(0.0004, 0.9, 0.999)
print("compiling")
model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
print("done")
for i in range(1,500):
	versus = 0
	changed = 0
	epsilon = 0.001
	last_kres = []
	sys.stdout.flush()
	print("i:", i)
	while(penew.hasNextBatch()):
		batchx, batchy, _ = penew.nextBacth()
		batchx = np.asarray(batchx)
		batchy = np.asarray(batchy)
		model.fit(batchx, batchy, nb_epoch=1, batch_size=50,class_weight=[0.999,0.001],verbose=0)
	penew.reset()

	while(peTest.hasNextBatch()):
		batchxT, batchyT, batchTestTable = peTest.nextBacth()
		batchxT = np.asarray(batchx)
		batchyT = np.asarray(batchy)
		system_predictions = model.predict(batchxT, batch_size=50, verbose=1)
		oracle_table = e.generate_table(batchTestTable, batchyT) 
		system_table = e.generate_table(batchTestTable, system_predictions) 
		recall_at_k_res = e.recall_at_k(oracle_table, system_table, k_s = [1,2,5,10,100,1000])
		print("recall_at_k_res: ", recall_at_k_res)
		
		# if(changed != 0):
			# versus = 0
			# for k in range(3):
				# if(recall_at_k_res[i] < last_kres[i] or ((recall_at_k_res[i] - last_kres[i]) != 0 and (recall_at_k_res[i] - last_kres[i]) < epsilon)):
					# versus = versus + 1
				# if(versus > 2):
					# K.set_value(optim.lr, 0.5 * K.get_value(optim.lr))
					# print("Changed lr value")
					# changed = 0
		# else:
			# changed = 1;
		
		# last_kres = []
		# for w in range(len(recall_at_k_res)):
			# last_kres.append(recall_at_k_res[w])	
		# print("last_kres: ", last_kres)
		
		peTest.reset()
    
