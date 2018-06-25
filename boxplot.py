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
      
def file_name(file_dir,choice): 
    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        # return dirs #当前路径下所有子目录  
        if choice == 0:
            return dirs
        else:
            return files #当前路径下所有非目录子文件


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
dirname = sys.argv[1]

server_flow = file_name(dirname, 1)
leng = int(len(server_flow))
for i in range(len(server_flow)):
    ffile = server_flow[i]
    reward = get_reward(dirname+'/'+ffile)
    # print(len(reward))
    # avg = np.mean(reward)
    flownum = ffile.replace('.txt','')
    datastore[flownum] = reward

data = pd.DataFrame(datastore)

data.boxplot()
data.boxplot()
plt.ylabel("Mean MOS Value")
plt.xlabel("Number of Flows")
plt.show()

