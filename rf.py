# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 20:50:11 2018

@author: 蔡欣强
"""

from math import log
from math import ceil
import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def ent(data):
    feat = {}
    for feature in data:
        curlabel = feature[-1]
        if curlabel not in feat:
            feat[curlabel] = 0
        feat[curlabel] += 1
    s = 0.0
    num = len(data)
    for it in feat:
        p = feat[it] * 1.0 / num
        s -= p * log(p,2)
    return s

def remove_feature(data,i,value,flag):
    newdata = []
    for row in data:
        if flag == True:
            if row[i] < value:
                temp = row[:i]
                temp.extend(row[i + 1:])
                newdata.append(temp)
        else:
            if row[i] >= value:
                temp = row[:i]
                temp.extend(row[i + 1:])
                newdata.append(temp)
#    print('newdata = ',newdata)
    return newdata

def choosebest(data):
    m = len(data)
    maxgain = 0.0
    bestfeature = -1
    bestpoint = -1.0
    n = len(data[0]) - 1
    S = ent(data)
    for i in range(n):
        curfeature = []
        for j in range(m):
            curfeature.append(data[j][i])
        curfeature = set(curfeature)
        curfeature = list(curfeature)
        maxgain = 0.0
        point_id = -1
        for j in range(len(curfeature) - 1):
            point = float(curfeature[j + 1] + curfeature[j]) / 2
            p1 = float(j + 1) / m
            p2 = float(m - j - 1) / m
            split = 0
            if p1 != 0:
                split -= p1 * log(p1,2)
            if p2 != 0:
                split -= p2 * log(p2,2)
            if split == 0:
                continue
            gain = (S - p1 * ent(remove_feature(data,i,point,True)) - p2 * ent(remove_feature(data,i,point,False))) / split
            if gain > maxgain:
                maxgain = gain
                bestfeature = i
                bestpoint = point
    return bestfeature,bestpoint
        
        

def classify(tree,feature,value):
    if type(tree).__name__ != 'dict':
        return tree
    root = list(tree.keys())[0]
    sons = tree[root]
    i = feature.index(root)
    if value[i] >= list(sons.keys())[1]:
        return classify(sons[list(sons.keys())[1]],feature,value)
    else:
        return classify(sons[list(sons.keys())[0]],feature,value)

def majorityCnt(classList):  
    classCount = {}  
    for vote in classList:  
        if vote not in classCount.keys(): classCount[vote] = 0  
        classCount[vote] += 1  
    sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]  


def build(data,feature):
    curlabel = [it[-1] for it in data]
#    print('cur data = ',data)
    if curlabel.count(curlabel[0]) == len(curlabel):
        return curlabel[0]
    if  len(data[0]) == 1:
        return majorityCnt(curlabel)
    i,point = choosebest(data)
#    print('i = ',i,'j = ',j)
    bestfeature = feature[i]
    tree = {bestfeature : {}}
    del feature[i]
    newfeature = feature[:]
    newdata = remove_feature(data,i,point,True)
    tree[bestfeature][0] = build(newdata,newfeature)
    newdata = remove_feature(data,i,point,False)
    newfeature = feature[:]
    tree[bestfeature][point] = build(newdata,newfeature)
    return tree
    


def dfs(tree,deep,sample):
    if (type(tree) != sample):
        return deep
    cnt = 0
    for key in tree.keys():
        cnt = max(cnt,dfs(tree[key],deep + 1,sample))
    return cnt

def get_subsample(data,ratio,label):
    data = np.array(data)
    Index = random.sample(list(range(len(data))),ceil(len(data) * ratio))
    subdata = data[Index]
    subdata = subdata.tolist()
    for i in range(len(Index)):
        subdata[i].append(label[Index[i]])
    return subdata

iris = load_iris()
x = iris['data']
y = iris['target']
feature = iris['feature_names']
data, test_data, label, ans = train_test_split(x,y, test_size = 0.3)
num = len(data)
rf = []
for i in range(5):
    subdata = get_subsample(data,0.7,label)
    print('subdata = ',subdata)
    test_feature = feature[:]
    rf.append(build(subdata,test_feature))
#for tree in rf:
#    print('tree = ',tree)
num = len(test_data)
res = []
for i in range(num):
    most_cnt = {}
    maxcnt = 0
    maxid = -1
    for tree in rf:
        Index = classify(tree,feature,test_data[i])
        if Index not in most_cnt:
            most_cnt[Index] = 1
        else:
            most_cnt[Index] += 1
        if most_cnt[Index] > maxcnt:
            maxcnt = most_cnt[Index]
            maxid = Index
#    print('most = ',most_cnt)
#    print('label = ',ans[i])
    res.append(Index)
cnt = 0
for i in range(num):
    if ans[i] == res[i]:
        cnt += 1
print('precise = ',cnt * 1.0 / num)
