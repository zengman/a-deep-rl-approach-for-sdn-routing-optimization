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
import os  
      
def file_name(file_dir): 

    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        # return dirs #当前路径下所有子目录  
        return files #当前路径下所有非目录子文件
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


# Dataflst = {}
datastore = dict()
datastore = OrderedDict()
for i in range(10):
    filename = (i + 1) * 5
    filename = str(filename)
    datastore[filename] = []
# print(datastore)
dirname = "rewardset"
server_flow = file_name(dirname)
leng = int(len(server_flow))
for i in range(len(server_flow)):
    ffile = server_flow[i]
    reward = get_reward(dirname+'/'+ffile)
    print(len(reward))
    # avg = np.mean(reward)
    flownum = ffile.replace('.txt','')
    
    # start = ffile.find('m_') + 2
    # end = len(ffile)
    # flownum =ffile[start:end]
    # print(flownum)
    datastore[flownum] = reward
data = pd.DataFrame(datastore)

#draw
data.boxplot()
plt.ylabel("Mean MOS Value")
plt.xlabel("Number of Flows")
plt.show()

