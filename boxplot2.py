# -*- coding:utf-8 -*-

"""
绘制箱体图
Created on 2017.09.04 by ForestNeo
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict  
"""
generate data from min to max
"""
import os  ,sys
      
def file_name(file_dir): 

    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        # return dirs #当前路径下所有子目录  
        return files #当前路径下所有非目录子文件

def file_dir(file_dir): 

    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        return dirs #当前路径下所有子目录  
        # return files #当前路径下所有非目录子文件
        # break
# df2 = pd.read_csv ("testfoo.csv" , encoding = "utf-8")
# print (df2)

def get_reward(filename):
    rewardlist = []
    # filename = 'rewardLog.txt'
    df = pd.read_csv(filename, header=None, sep=',')
    a = np.asarray(df) 
    for x in a:
        rewardlist.append(x[0])
    # print(rewardlist)
    return rewardlist

def get_ut_reward(atype,namelist, type):
    rewardlist = []
    for i in range(len(namelist)):
        # foldername = folder + str(i) + '_' + str(atype) + '/'
        foldername = mainfolder + namelist[i] + '/'
        if type in foldername:
            df = pd.read_csv(foldername+filename, header=None, sep=',')
            rewardlist.append(df.head(1))
    return np.asarray(rewardlist)
# Dataflst = {}
datastore = dict()
datastore = OrderedDict()
datastoreU = dict()
datastoreU = OrderedDict()
datastoreT = dict()
datastoreT = OrderedDict()
for i in range(10):
    filename = (i + 1) * 5
    filename = str(filename)
    datastore[filename] = []
# print(datastore)
dirname = sys.argv[1]
server_flow = file_name(dirname)
leng = int(len(server_flow))
for i in range(len(server_flow)):
    ffile = server_flow[i]
    reward = get_reward(dirname+'/'+ffile)
    # print(len(reward))
    # avg = np.mean(reward)
    flownum = ffile.replace('.txt','')
    datastore[flownum] = reward
    # datastore2[flownum] = reward 
    # datastore2[flownum][0] += 0.01
# utdir = sys.argv[2]
# server_flow = file_dir(utdir)
# leng = int(len(server_flow))
# for x in server_flow:
#     flowlist = file_dir(x)
#     for y in flowlist:
#         if 'ttt' in y:
#             reward = get_ut_reward('u',y, 'tttt')
#             s = y.find('_') + 1
#             e = y.find('_tttt') 
#             datastoreT[y[s:e]] = reward
#         else:
#             reward = get_ut_reward('t', y, 'uuuu')
        




# print(datastore)
# dirname = "reward2"
# server_flow = file_name(dirname)
# leng = int(len(server_flow))
# for i in range(len(server_flow)):
#     ffile = server_flow[i]
#     if 'zip' in ffile:
#         continue
#     reward = get_reward(dirname+'/'+ffile)
#     flownum = ffile.replace('.txt','')
#     datastore[flownum] = reward

# dirname = "reset"
# server_flow = file_name(dirname)
# leng = int(len(server_flow))
# for i in range(len(server_flow)):
#     ffile = server_flow[i]
#     if 'zip' in ffile:
#         continue
#     reward = get_reward(dirname+'/'+ffile)
#     flownum = ffile.replace('.txt','')
#     datastore[flownum] = reward




data = pd.DataFrame(datastore)

# data2 = pd.DataFrame(datastore2)

data.boxplot()
# data2.boxplot(boxprops = {'color':'black'})
plt.ylabel("Mean MOS Value")
plt.xlabel("Number of Flows")
plt.show()

