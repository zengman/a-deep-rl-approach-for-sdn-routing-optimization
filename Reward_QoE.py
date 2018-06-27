import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
inputfile = 'NN.xlsx'   #excel输入
def NN_training():
    #1 数据输入
    global inputfile
    # inputfile = 'NN.xlsx'   #excel输入
    # outputfile = 'output.xls' #excel输出
    modelfile = 'modelweight.model' #神经网络权重保存
    data = pd.read_excel(inputfile,index='Date',sheet_name=0) #pandas以DataFrame的格式读入excel表
    feature = ['F1','F2','F3','F4'] #影响因素四个
    label = ['L1'] #标签一个，即需要进行预测的值
    # row_len = data.index
    # print(row_len)
    
    # data_train = data.loc[range(0,3000)].copy() #标明excel表从第0行到520行是训练集3000
    data_train = data.copy()

    #2 数据预处理和标注
    data_mean = data_train.mean()  
    data_std = data_train.std()  
    data_train = (data_train - data_mean)/data_std #数据标准化
    x_train = data_train[feature].as_matrix() #特征数据
    y_train = data_train[label].as_matrix() #标签数据

    #3 建立一个简单BP神经网络模型
    model = Sequential()  #层次模型
    model.add(Dense(12,input_dim=4,init='uniform')) # 输入层，Dense表示BP层
    model.add(Activation('relu'))  #添加激活函数
    model.add(Dense(12,input_dim=12,init='uniform')) #输入层，Dense表示BP层
    model.add(Activation('relu'))  #添加激活函数
    model.add(Dense(1,input_dim=12))  #输出层
    model.compile(loss='mean_squared_error', optimizer='sgd')
    # optimizer='adma') #编译模型
    model.fit(x_train, y_train, nb_epoch = 600, batch_size = 32) #训练模型1000次
    model.save_weights(modelfile) #保存模型权重
    # model.save_weights()
    return model, data_mean, data_std

def testmodel():
    global inputfile
    modelfile = 'modelweight.model' #神经网络权重保存
    model = Sequential()  #层次模型
    model.add(Dense(12,input_dim=4,init='uniform')) # 输入层，Dense表示BP层
    model.add(Activation('relu'))  #添加激活函数
    model.add(Dense(12,input_dim=12,init='uniform')) #输入层，Dense表示BP层
    model.add(Activation('relu'))  #添加激活函数
    model.add(Dense(1,input_dim=12))  #输出层
    model.compile(loss='mean_squared_error', optimizer='sgd')
    model.load_weights(modelfile)

    data = pd.read_excel(inputfile,index='Date',sheet_name=0) #pandas以DataFrame的格式读入excel表
    # data_train = data.loc[range(0,3000)].copy() #标明excel表从第0行到520行是训练集3000
    data_train = data.copy()

    #2 数据预处理和标注
    data_mean = data_train.mean()  
    data_std = data_train.std()  

    return model, data_mean, data_std


def NN_pridict(data, model, data_mean, data_std):
    feature = ['F1','F2','F3','F4'] 
    #4 预测，并还原结果。
    x = ((data[feature] - data_mean[feature])/data_std[feature]).as_matrix()
    value = np.full((x[0][0].shape[0],4), -1, dtype=float)
    for i in range(4):
        value[:,i] = x[0][i]
    y = model.predict(value) * data_std['L1'] + data_mean['L1']
    return y


def lossreward(loss, reward):
    for i in range(len(loss)):
        lo = loss[i]
        if lo >= 0.4:
            reward[i] -= lo * 0.45
        else:
            reward[i] -= lo * 0.01
    return reward

def reward_QoE(x1,x2,x3,x4, model, data_mean, data_std):
    print('band:')
    print(x1)
    print('delay:')
    print(x2)
    print('jitter:')
    print(x3)
    print('loss:')
    print(x4)
    feature = ['F1','F2','F3','F4'] 
    # model, data_mean, data_std = NN_training()
    b=[]
    b.append(x1)
    b.append(x2)
    b.append(x3)
    b.append(x4)
    data_state = pd.DataFrame([b], columns= feature, index=['0'])
    reward = NN_pridict(data_state, model, data_mean, data_std)
    # reward = lossreward(x4, reward)
    print('reward')
    print(reward)
    # zhognweishu = np.median(reward)
    # print('中位数:=', zhognweishu)

    print('均值', np.mean(reward))

    # result = np.mean(reward)
    # if result>5:
    #     result = 5
    # print("Rewad = " + str(result))
    # result = (result - 5) * 1000
    # print(np.mean(reward))
    # return reward.sum() - 41
    return np.mean(reward)
    
#     #5 导出结果
#     data.to_excel(outputfile) 

# #6 画出预测结果图
# import matplotlib.pyplot as plt 
# p = data[['L1','L1_pred']].plot(subplots = True, style=['b-o','r-*'])
# plt.show()

# NN_training()