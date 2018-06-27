#!python3
# -*- coding:utf-8 -*-
import math
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split


# Tool Fuction group
class FuncTool:
    '''
        Here we'll define some useful function that will be applied 
     to the following preprocessing.
    '''
    ####################################################################
    # If you'd like to add some other functions, you can do like this: #
    #   def func(self,*args):                                          #
    #       pass                                                       #
    # And call it in other child class by: self.func                   #
    ####################################################################
    def __init__(self):
        self.needclean = []
        self.addColList = []
        self.dict2 = {}

    def needClean(self, fea,col):
        #Func
        #   ----------------
        #   To judge whether we need to clean this feature or not.

        #Param
        #   ----------------
        #   fea: name of the feature
        #   col: pandas.Series, the column of feature.

        #Return
        #   ----------------
        #   boolean: True or False
        #
        if type(col[0]).__name__ == 'str':
            # Judge whether requires transfromation
            if np.sum(col.apply(lambda x:x[-1]==')')*1) != 0:
                self.needclean.append(fea)
                
                return True 

        return False


    # Tool Function applied to columns of dataframe
    # Replace the error with ground truth
    def correct(self, value):
        # To make sure if there needs to be corrected.
        if value.endswith(')'):
            # continuous type or discrete type
            if value[0].isdigit():
                l = re.findall(r'\d+',value)

                return int(l[1])
            else:
                v = value[:-1].split('(')
                value = v[-1].split(',')

                return value[0]

        return value

    # Tool Function applied to columns of dataframe
    def number(self, value):
        # To make sure if there needs to be corrected.
        if value.endswith(')'):
            # continuous type or discrete type
            if value[0].isdigit():
                l = re.findall(r'\d+',value)

                return int(l[2])
            else:
                v = value[:-1].split('(')
                value = v[-1].split(',')

                return value[1]

        return 0


# Main method to process dataset containing split and transformation.
class DataClean(FuncTool):
    '''
        Dataset should be type of dataframe. See more detail in demo.ipynb.
    '''

    # The way we select samples and split train and test.
    def split(self, data, K, split_prob=0.3):
        '''
        #Param
        #   ----------------
        #      dataset: pandas.DataFrame (whole dataset);
        #            K: int or float, given to tell the number or rate of samples;
        #   split_prob: float, given to tell the size of test_set.
        #
        #Output
        #   ----------------
        #   X_train,X_test,y_train,y_test: pandas.DataFrame. 
        '''
        if K<1:
            sample_set = data.sample(frac=K, axis=0)
            print('{:.2f} % will be selected as sample set.'.format(K*100))
        else:
            sample_set = data.sample(n=K, axis=0)
            print('{} rows will be selected as sample set.'.format(K))

        print('-'*30)
        print('{} rows will be selected as train set.'.format(math.floor(K*(1-split_prob))))
        print('-'*30)

        value = sample_set[sample_set.columns[:-1]]
        label = sample_set[sample_set.columns[-1]]

        return train_test_split(value, label, test_size=split_prob, random_state=0)

    def transform(self, X):
        '''
        #Param
        #   -----------------
        #   X: pandas.DataFrame, value of the sampleset.
        #
        #Output
        #   -----------------
        #   cleandata: pandas.DataFrame, X after processing.
        '''
        # Make a copy. We should always avoid to processing on the origin dataset.
        Xclean = X[:]

        # First we need to judge whether the feature needs clean.
        # If it requires, we correct mistake in the copy.
        # And then create a new column in copy to record the repeat number.
        print('Now Start Transformation!')
        print('-'*30)
        for fea in X.columns:
            if self.needClean(fea, X[fea]):
                Xclean[fea] = X[fea].apply(self.correct)
                Xclean[fea+'_repnum'] = X[fea].apply(self.number)
                
                self.addColList.append(fea+'_repnum')

        for fea in X.columns[1:]:
            if type(Xclean.loc[0,fea]).__name__ == 'int64':
                n = X.shape[0]
                sigma_number = np.sum(1/Xclean[fea+'_repnum'])

                d = n/sigma_number
                Xclean[fea] = (Xclean[fea]*d) / Xclean[fea+'_repnum']

        return Xclean[X.columns]







