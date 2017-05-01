from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Merge
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

base = "../standard_split_reduced_3/"

X_train,X_train_specific_features,Y_train,max_value,_,Pairs = cr.read_corpus(base+"Dev-training.txt", base+"blocks.txt", 3,n_of_words=max_num_of_words)

X_test,X_test_specific_features,Y_test,max1,Y_test_tables,PairsTest = cr.read_corpus(base+"Dev-testing_full_grouped.txt", base+"blocks.txt", 3,max_lenght_of=max_value,n_of_words=max_num_of_words)


penew = pe.PreEmbedderNew(bp.Concatenation, Pairs, Y_train, X_train_specific_features, batchDim = 200)

peTest = pe.PreEmbedderNew(bp.Concatenation, PairsTest, Y_test, X_test_specific_features, Y_test_table = Y_test_tables, batchDim = 200)


blocks_model = Sequential()

feature_model = Sequential()

final_model = Sequential()


blocks_model.add(Dense(input_dim=8192,output_dim=300, activation='relu'))
blocks_model.add(Dropout(0.5))
blocks_model.add(Dense(300, activation='relu'))

feature_model.add(Dense(input_dim=9, output_dim=300, activation='relu'))
feature_model.add(Dropout(0.5))
feature_model.add(Dense(300, activation='relu'))


final_model.add(Merge([blocks_model, feature_model], mode='concat', concat_axis=-1))
final_model.add(Dense(output_dim=2))
final_model.add(Activation("relu"))
final_model.add(Activation("softmax"))

print("compiling")
optim = opt.adam(0.0004, 0.9, 0.999)
final_model.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])
print("done")

global_k_res = []
for j in range(6):
	global_k_res.append(0)


for i in range(1,500):
	a = 0
	versus = 0
	changed = 0
	epsilon = 0.001
	last_kres = []
	sys.stdout.flush()
	print("i:", i)
	while(penew.hasNextBatch()):
		batchx, batchy, _, featuresTrain = penew.nextBacth()
		batchx = np.asarray(batchx)
		batchy = np.asarray(batchy)
		featuresTrain = np.asarray(featuresTrain)
		final_model.fit([batchx, featuresTrain], batchy, nb_epoch=1, batch_size=50,class_weight=[0.999,0.001],verbose=0)
	penew.reset()

	while(peTest.hasNextBatch()):
		print(i, " ", a)
		batchxT, batchyT, batchTestTable, featuresTest = peTest.nextBacth()
		batchxT = np.asarray(batchxT)
		batchyT = np.asarray(batchyT)
		featuresTest = np.asarray(featuresTest)
		system_predictions = final_model.predict([batchxT, featuresTest], batch_size=50, verbose=0)
		oracle_table = e.generate_table(batchTestTable, batchyT) 
		system_table = e.generate_table(batchTestTable, system_predictions) 
		recall_at_k_res = e.recall_at_k(oracle_table, system_table, k_s = [1,2,5,10,100,1000])
		print("recall_at_k_res: ", recall_at_k_res)
		for m in range(len(recall_at_k_res)):
			if recall_at_k_res[m] > global_k_res[m]:
				global_k_res[m] = recall_at_k_res[m]
	peTest.reset()

print("Max_k_res: ", global_k_res)


