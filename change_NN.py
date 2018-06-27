import pandas as pd
# from keras.models import Sequential
# from keras.layers.core import Dense, Activation
import numpy as np

def NN_change():
    #1 数据输入
    inputfile = 'NN.xlsx'   #excel输入
    # outputfile = 'output.xls' #excel输出
    # modelfile = 'modelweight2.model' #神经网络权重保存
    data = pd.read_excel(inputfile,index='Date',sheet_name=0) #pandas以DataFrame的格式读入excel表
    feature = ['F1','F2','F3','F4'] #影响因素四个
    label = ['L1'] #标签一个，即需要进行预测的值
    # print(data[:,4])
    print(data)

NN_change()

   