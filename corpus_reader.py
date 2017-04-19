'''
Created on 26/set/2016

@author: Fabio Massimo Zanzotto
'''

#import resource
import keras.preprocessing.text as p
import keras.preprocessing.sequence as seq
import sys
import numpy as np
import statistics as stat
from blockpair import *


def localpad_sequences(word_sequences , maxlen=None):
    if maxlen == None:
        lens = [len(a) for a in word_sequences]
        maxlen = np.max(lens)
        avglen = stat.mean(lens)
        stand_dev = np.sqrt(stat.variance(lens, avglen)) 
        print("Stats ", maxlen,avglen,stand_dev)
        maxlen = np.int(avglen + stand_dev)
    #NOWIN print("Mem Used (2) ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    print("Expected Occupancy = " , maxlen, len(word_sequences), maxlen*len(word_sequences)*4/1024)
    
    word_sequences = seq.pad_sequences(word_sequences,maxlen=maxlen,truncating="pre")
    
    
    #for idx in range(0,len(word_sequences)):
    #for idx,s in enumerate(word_sequences):
        #print("Mem Used (3) ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    #    out = np.zeros(maxlen)
    #    out[(maxlen-len(word_sequences[idx])):maxlen] = word_sequences[idx] 
    #    word_sequences[idx] = out.tolist()
    return word_sequences



def read_block_file(file_name,block_lenght):
    blocks = {}
    #print("Encoding : ", sys.getdefaultencoding())
    bf = open(file_name,"r",encoding="utf-8")
    description_l = bf.readline()
    description = description_l.split("\t")[1:]
    for line in bf.readlines():
        blocks[line.split("\t")[0]] = dict(zip(description,line.split("\t")[1:]))
    bf.close()
    return (description,blocks)


def encode_label(l):
    return [1,0] if l == "1" else [0,1]
#   return 1 if l == "1" else 0

def separator(n):
    return (abs(hash("#Separator#")) % (n - 1) + 1)

def read_corpus(corpus_file,block_file,block_lenght,n_of_words = 40000,max_lenght_of=None):
    print("Reading corpus : " , corpus_file)
    (_,blocks) = read_block_file(block_file, block_lenght)
    bf = open(corpus_file,"r",encoding="utf-8")
    bf.readline()
    X = []
    X_specific_features = []
    Y = []
    Y_labels = []
    blocks_for_PreEmbedder = []
    Pairs = []
    words = {}
    for line in bf.readlines():
        BP = BlockPair()
        (block_a_text,block_a_features,speakers_a) = extract_block_features(blocks[line.split(":")[1]],block_lenght,n_of_words)
        (block_b_text,block_b_features,speakers_b) = extract_block_features(blocks[line.split(":")[2].strip()],block_lenght,n_of_words)
        for key,value in blocks[line.split(":")[1]].items():
            if(key[:6] == "syntax"):
                BP.addTreeToBlockOne(value)
        for key, value in blocks[line.split(":")[2].strip()].items():
            if(key[:6] == "syntax"):
                BP.addTreeToBlockTwo(value)
        X.append(block_a_text + [separator(n_of_words)] + block_b_text)
        Pairs.append(BP)
        group_features = [
            len(set(speakers_a).intersection(set(speakers_b)))/len(set(speakers_a).union(set(speakers_b))),
            len(set(speakers_a).intersection(set(speakers_b)))/len(set(speakers_a)),
            len(set(speakers_a).intersection(set(speakers_b)))/len(set(speakers_b))]
        if np.linalg.norm(group_features) > 0:
            group_features = group_features/np.linalg.norm(group_features)
        stylistic_features = block_a_features + block_b_features
        if np.linalg.norm(stylistic_features) > 0:
            stylistic_features = stylistic_features/np.linalg.norm(stylistic_features)
        X_specific_features.append(  np.concatenate([group_features,stylistic_features]).tolist() ) 
        Y.append(encode_label(line.split(":")[0]))
        Y_labels.append((line.split(":")[1],line.split(":")[2]))
    print("X length ", len(X))
    #NOWIN print("Mem Used ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    #X = seq.pad_sequences(X,maxlen=max_lenght_of)
    X = localpad_sequences(X,maxlen=max_lenght_of)
    #NOWIN print("Padded Mem Used ", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bf.close()
    max_len = 0
    if len(X) > 0:
        max_len = len(X[0])
    '''   
    indexes = np.random.random_integers(0,len(X_specific_features),10)   
    for i in indexes:
        print("X FEATURES: ",X_specific_features[i])
        print("X :", X[i])
    '''
    
    
    return X,X_specific_features,Y, max_len ,Y_labels, Pairs
    


def extract_block_features(block,block_lenght,n_of_words):
    block_a_text = []
    speakers = []
    for i in range(1,block_lenght + 1):
        l = 0 
        len_in_words = 0
        if "turn_" + str(i) in block:
            l = l + len(block["turn_" + str(i)])
            turn_words = p.one_hot(block["turn_" + str(i)],n_of_words)
            len_in_words = len_in_words + len(turn_words)
            block_a_text = block_a_text + turn_words
            speakers.append(block["character_" + str(i)])
    block_a_features = [l/block_lenght if block_lenght > 0 else 0,len_in_words/block_lenght if block_lenght > 0 else 0,l/len_in_words if len_in_words > 0 else 0]
    return (block_a_text,block_a_features,speakers)

#file = "C:\\USER_DATA\\FABIO\\LAVORO\\PROGETTI\\SAG_SVN\\LatentPlotRecognition\\ReleasedCorpus\\standard_split_reduced_5\\blocks.txt"
#read_block_file(file,3)



