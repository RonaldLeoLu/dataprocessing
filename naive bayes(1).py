# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:06:39 2017

@author: cwd
"""

#coding:utf-8
#朴素贝叶斯算法
import pandas as pd
import numpy as np
import math
#import json
import pickle as p

class NaiveBayesB(object):
    def getTrainSet(self):
        print('gettrain_begin')
        labels_D_num=1    #分类属性个数
#        trainSet = pd.read_csv('E://code//bayes-python//naivebayes_data.csv',encoding='gb2312')
        trainSet = pd.read_csv('D:\\dataset\\adult_train_1.csv',encoding='gb2312')
        
        trainSetNP = np.array(trainSet)     #由dataframe类型转换为数组类型
        trainData = trainSetNP[:,0:trainSetNP.shape[1]-labels_D_num]     #训练数据x1,x2
        labels = trainSetNP[:,trainSetNP.shape[1]-labels_D_num:trainSetNP.shape[1]]          #训练数据所对应的所属类型Y
        print("gettrainset_end")
        return trainData, labels


    def parameter(self,trainData,labels):
        print("parameter")
        
        self.Lanmda = 1                             #平滑参数   “1”为拉普拉斯平滑
        self.trainData_deduplication=[]             #特征属性去重
        self.labels_deduplication=[]                #分类属性去重
        self.trainData_num=[]                       #特征属性非重复值个数
        self.labels_num=[]                          #分类属性非重复值个数
        self.trainData_count=[]                     #特征属性列数
        self.labels_count=[]                        #分类属性列数
        
        a=[]
        self.trainData_num=trainData.shape[1]
        self.labels_num=labels.shape[1]
        for i in range(self.trainData_num):
            a=list(set(trainData[:,i]))
            self.trainData_count.append(len(a))        #统计异值个数
            self.trainData_deduplication.append(a)   #特征属性去重
        for i in range(self.labels_num):
            a=list(set(labels[:,i]))
            self.labels_count.append(len(a))
            self.labels_deduplication.append(a)         #分类属性去重
		
        return 0

    def bayes_Model(self, trainData, labels,continuos):
        print('model_train_begin')


     
        #求先验概率
        k=0
        P_y = {}
#        P_y_log = {}
        for label in nb.labels_deduplication[0]:   #"0"表示类别属性第一列，多列类别属性暂不考虑
            labels = list(labels)    #转换为list类型
            P_y[label] = (labels.count(label) + nb.Lanmda) / float(len(labels) + nb.Lanmda*nb.labels_count[0])
#            P_y_log[label]=math.log(P_y[label])


        #求条件概率
        P = {}
#        P_log = {}
        for y in nb.labels_deduplication[0]:
            y_index = [i for i, label in enumerate(labels) if label == y]   # y在labels中的所有下标
            y_count = labels.count(y)     # y在labels中出现的次数
            for j in range(nb.trainData_num):
                dict1 = {}
#                dict1_log = {}
                if j in continuous:
                    sum_=0
                    deviation_ = 0
                    for i in y_index:
                        sum_ += trainData[i,j]
                    mean_ = sum_ * 1. / y_count
                    dict1['mean|'+str(y)]=mean_
                    for i in y_index:
                        deviation_2 = trainData[i,j] - mean_
                        deviation_ += deviation_2 * deviation_2
                    var_ = deviation_ / (y_count-1)#极大似然估计得到均值和方差参数
                    dict1['var|'+str(y)]=var_
                else:
                    for a in nb.trainData_deduplication[j]:
                        pkey = str(a) + '|' + str(y)
                        xy_count = 0
                        for k in y_index:
                            if (trainData[k,j] == a):
                                xy_count+= 1           #x y同时出现的次数
                            dict1[pkey] = (xy_count + nb.Lanmda) / float(y_count + nb.Lanmda*nb.trainData_count[j])   #条件概率
#                        dict1_log[pkey] = math.log(dict1[pkey])
                if j in P.keys():
                    P[j].update(dict1)
#                    P_log[j].update(dict1_log)
                else:
                    P[j] = dict1
#                    P_log[j] = dict1_log 
    
        
        filePath1="D:\\dataset\\P_y.pkl"
        filePath2="D:\\dataset\\P.pkl"
# =============================================================================
#         filePath3="E://code//bayes-python/model//P_y_log.pkl"
#         filePath4="E://code//bayes-python//model//P_log.pkl"
#         
# =============================================================================
        f1=open(filePath1,"wb")
        p.dump(P_y,f1)
        f1.close()
        
        f2=open(filePath2,"wb")
        p.dump(P,f2)
        f2.close()

# =============================================================================
#         f3=open(filePath3,"wb")
#         p.dump(P_y_log,f3)
#         f3.close()
# 
#         f4=open(filePath4,"wb")
#         p.dump(P_log,f4)
#         f4.close()
#         
# =============================================================================
        print('model_train_end')
        return P,P_y
                  
    def classify(self,lables,testData,continuous):
        print('classify_beging')
        
        filePath1="D:\\dataset\\P_y.pkl"
        filePath2="D:\\dataset\\P.pkl"
# =============================================================================
#         filePath3="E://code//bayes-python//model//P_y_log.pkl"
#         filePath4="E://code//bayes-python//model//P_log.pkl"
# 
# =============================================================================
        f1=open(filePath1,"rb")
        P_y = p.load(f1)
        f1.close()
        
        f2=open(filePath2,"rb")
        P = p.load(f2)
        f2.close()
        
# =============================================================================
#         f3=open(filePath3,"rb")
#         P_y_log = p.load(f3)
#         f3.close() 
#         
#         f4=open(filePath4,"rb")
#         P_log = p.load(f4)
#         f4.close()
#         
# =============================================================================
        P_pos = {}
#        P_pos_log = {}
        classical = {}
        for i in range(len(testData)):
            P_x=0.
#            P_x_log=0.
            
            P_pos[i] = {}
#            P_pos_log[i] = {}

            for y in self.labels_deduplication[0]:
                a= P_y[y]
#                aa=P_y_log[y]
                k=0
                pkey = str(y) + '|' 
                for j in range(self.trainData_num):
#                    print(j)
                    if j in continuous:
                        b = 1. / (math.sqrt(2 * math.pi * P[j][('var|' + str(y))])) * math.exp(-(testData[i,j] - P[j][('mean|' + str(y))])**2 / (2 * P[j][('var|' + str(y))]))
                    else:
                        if (testData[i][j] in self.trainData_deduplication[k]):
                            b = P[j][str(testData[i][j])+'|'+str(y)]
#                            bb = P_log[j][str(testData[i][j])+'|'+str(y)]
                        else:
                            b = 1/100                    
#                            bb = math.log(1/100)
                    a*= b
#                    aa+= bb
                    k+= 1
                    pkey = pkey  + str(testData[i][j])
                P_pos[i][pkey] = a
#                P_pos_log[i][pkey] = aa
#                print(P_pos)
                P_x += a
#                P_x_log +=aa
                
                prop=0
            for y in self.labels_deduplication[0]:
                k=0
                pkey = str(y) + '|' 
                for x in testData[i]:
                    k+= 1
                    pkey = pkey  + str(x)
                if(P_pos[i][pkey] > prop):
                    a = y
                    prop = P_pos[i][pkey]
                P_pos[i][pkey] = P_pos[i][pkey] / P_x
#                P_pos_log[i][pkey] = P_pos_log[i][pkey] - P_x_log
#            print(a)
            classical[i] = a
                
        print('classicify_end')
        return P_pos,classical
    
    def evaluate(self,classical,test_label):
        count=0
        for i in range(len(test_label)):
            if classical[i] == test_label[i]:
                count+=1
        precision=count/len(test_label)
        print(precision)

if __name__ == '__main__':
    nb = NavieBayesB()
    # 训练数据
    continuous = [1]
    trainData, labels = nb.getTrainSet()
    testData_NP = np.array(pd.read_csv('D:\\dataset\\adult_test_1.csv',encoding='gb2312'))
    testData = testData_NP[:,0:testData_NP.shape[1]-1]     #训练数据x1,x2 
    test_label = testData_NP[:,testData_NP.shape[1]-1]
#    testData = trainData
#    test_label = labels
#    testData = np.array([[2,'S']])
    nb.parameter(trainData,labels)
    P,P_y = nb.bayes_Model(trainData,labels,continuous)
    L,classical = nb.classify(labels,testData,continuous)
    # 该特征应属于哪一类
    nb.evaluate(classical,test_label)
    
#    print ('后验概率:',json.dumps(L,indent=5,ensure_ascii=False))
#    print ('后验概率log:',json.dumps(L_log,indent=5,ensure_ascii=False))
#    print ('分类：',json.dumps(classical,indent=5,ensure_ascii=False))
    

# =============================================================================
# for i in range(len(testData[:,0])):
#     a+ = (testData[i,0]-np.mean(testData[:,0])*(testData[i,0]-np.mean(testData[:,0]
# =============================================================================
