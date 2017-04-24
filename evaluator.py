'''
Created on 26/set/2016

@author: Fabio Massimo Zanzotto
'''

import numpy


def recall_at_k_singlepoint(oracle,system,k_s = [1,2,5,10]):
    o_s = zip(oracle,system)
    o_s = sorted(o_s,key=lambda x:x[1],reverse=True)
    #print("Recall at k = " , o_s[0:9])
    res = numpy.zeros(len(k_s))
    for i in range(0,min(max(k_s),len(o_s))):
        (scelta,_) = o_s[i]
        if scelta == 1.0:
            for j in range(0,len(k_s)): 
                if i < k_s[j]:
                    res[j] = 1
    return res


def read_file(file):
    oracle = open(file,"r",encoding="utf-8")
    oracle.readline()
    oracle_table = {}
    for l in oracle.readlines():
        if l.split(":")[1] not in oracle_table:
            oracle_table[l.split(":")[1]] = []
        oracle_table[l.split(":")[1]].append((float(l.split(":")[0]),l.split(":")[2]))
    oracle.close()
    return oracle_table


def generate_table(table_of_ids,results):
    oracle_table = {}
    for ((r,_),(id_start,id_end)) in zip(results,table_of_ids):
        if id_start not in oracle_table:
            oracle_table[id_start] = []
        oracle_table[id_start].append((float(r),id_end))
    return oracle_table


def recall_at_k_with_files(oracle_file,system_file, k_s = [1,2,5,10]):
    return recall_at_k(read_file(oracle_file),read_file(system_file), k_s)

def recall_at_k(oracle_table,system_table, k_s = [1,2,5,10]):
    recall_at_k_global = numpy.zeros(len(k_s))
    for k in oracle_table:
        oracle = [ o for (o,_) in oracle_table[k]]
        oracle_tags = [ o for (_,o) in oracle_table[k]]
        system = [ o for (o,_) in system_table[k]]
        system_tags = [ o for (_,o) in system_table[k]]
        if not oracle_tags == system_tags :
            print(" EEEE  ", oracle_tags ,"\n    ", system_tags)
        partial = recall_at_k_singlepoint(oracle,system,k_s)
        for i in range(0,len(k_s)):
            recall_at_k_global[i] = recall_at_k_global[i] + partial[i]
    size = len(oracle_table.keys())
    for i in range(0,len(k_s)):
        recall_at_k_global[i] = recall_at_k_global[i]/size
    return recall_at_k_global

##### main

#base = "C:\\USER_DATA\\FABIO\\LAVORO\\PROGETTI\\SAG_SVN\\LatentPlotRecognition\\temp\\"

#oracle = base + "oracle.txt"
#system = base + "system.txt"

#recall_at_k_global = recall_at_k_with_files(oracle , system , [1,2,5,10])

#print("Results = " , recall_at_k_global)








