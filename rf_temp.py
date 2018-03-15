# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 18:54:11 2018

@author: jkrs
"""

from math import log
import matplotlib.pyplot as plt
import random
import operator
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

def remove_feature(data,i,split_point,value,flag):
    newdata = []
#    print('split_point = ',split_point)
    num = 0
    n = len(data[0]) - 1
    for j in range(len(data)):
        if flag == True:
            if num == split_point:
                break
            if data[j][i] <= value:
                num += 1
                temp = data[j][:i]
                temp.extend(data[j][i + 1:])
                newdata.append(temp)
        else:
            if num == n - split_point:
                break
            if data[j][i] >= value:
                num += 1
                temp = data[j][:i]
                temp.extend(data[j][i + 1:])
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
    split_num = -2
    for i in range(n):
        curfeature = []
        for j in range(m):
            curfeature.append(data[j][i])
        curfeature.sort()
        maxgain = 0.0
        point_id = -1
        for j in range(m - 1):
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
            gain = (S - p1 * ent(remove_feature(data,i,j + 1,point,True)) - p2 * ent(remove_feature(data,i,j + 1,point,False))) / split
            if gain > maxgain:
                maxgain = gain
                bestfeature = i
                bestpoint = point
                split_num = j
    return bestfeature,split_num + 1,bestpoint
        
        

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
    i,j,point = choosebest(data)
#    print('i = ',i,'j = ',j)
    bestfeature = feature[i]
    tree = {bestfeature : {}}
    del feature[i]
    newfeature = feature[:]
    newdata = remove_feature(data,i,j,point,True)
    tree[bestfeature][0] = build(newdata,newfeature)
    newdata = remove_feature(data,i,j,point,False)
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

def main():
    minn = 100
    maxn = -1
    iris = load_iris()
    x = iris['data']
    y = iris['target']
    feature = iris['feature_names']
    result = []
    for step in range(1):
        data, test_data, label, ans = train_test_split(x,y, test_size = 0.3)
        data = data.tolist()
        label = label.tolist()
        num = len(data)
        for i in range(num):
            data[i].append(label[i])
    #    print('data = ',data)
        test_feature = feature[:]
        tree = build(data,test_feature)
        test_data = test_data.tolist()
        num = len(test_data)
        res = []
        for i in range(num):
            res.append(classify(tree,feature,test_data[i]))
        cnt = 0
        ans = ans.tolist()
        for i in range(num):
            if ans[i] == res[i]:
                cnt += 1
    #    print('precise = ',cnt * 1.0 / num)
        result.append(cnt * 1.0 / num)
        maxn = max(maxn,cnt * 1.0 / num)
        minn = min(minn,cnt * 1.0 / num)
    print('maxn = ',maxn,'minn = ',minn)
    print(result)

if __name__ == '__main__':
    main()
