# -*- coding:utf-8 -*-

"""
绘制箱体图
Created on 2017.09.04 by ForestNeo
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict  
import seaborn as sns
import os  ,sys
"""
generate data from min to max
"""

def file_name(file_dir): 

    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        # return dirs #当前路径下所有子目录  
        return files #当前路径下所有非目录子文件

def file_dir(file_dir): 
    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        return dirs #当前路径下所有子目录  
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
    
def get_ut_reward(folder,data):
    dirs = file_dir(folder)
    if dirs == None:
        print(folder)
    # print(dirs)
    for y in dirs:
        reward = get_reward(folder + '/' + y + '/rewardLog.txt')
        reward = np.mean(reward)
        data['reward'].append(reward)
        s = y.find('_') + 1
        if 'tttt' in y:
            e = y.find('_tttt') 
            flownum = y[s:e]
            data['flow_num'].append(flownum)
            data['type'].append('Throughout')
            
        else:
            e = y.find('_uuuu')
            flownum = y[s:e]
            data['flow_num'].append(flownum)
            data['type'].append('Utility')
    # ureward['5'] = 4.8
    # return ureward.copy(), treward.copy()

# ut reward draw
def utrewarddata(data):
    storefolder = './matUT/'
    for i in range(18):
        index = i + 1
        matfile = storefolder +  'mat' + str(index) + '_1'
        get_ut_reward(matfile,data)

def drlreward(data,dirname):
    
    server_flow = file_name(dirname)
    leng = int(len(server_flow))
    for i in range(len(server_flow)):
        ffile = server_flow[i]
        reward = get_reward(dirname+'/'+ffile)
        flownum = ffile.replace('.txt','')
        for r in reward:
            data['flow_num'].append(flownum)
            data['type'].append('DRL')
            data['reward'].append(r)

# utrewarddata(datastoreT,datastoreU)
data= OrderedDict()
data['flow_num'] = []
data['type'] = []
data['reward'] = []
# print(datastore)
dirname = sys.argv[1]
drlreward(data, dirname)
# utrewarddata(data)
# get_ut_reward('Mat03',data)
# print(data)
sns.set(style="ticks")
save = pd.DataFrame(data,columns=['flow_num','type','reward'])
save.to_csv('rewardlist.csv',sep=',',index = False)
tips = pd.read_csv("rewardlist.csv")
# Draw a nested boxplot to show bills by day and sex
sns.boxplot(x="flow_num", y="reward", hue="type", data=tips, palette="PRGn")
sns.despine(offset=10, trim=True)
# plt.figure(figsize=(20, 6))
plt.show()

# plt.ylabel("Mean MOS Value")
# plt.xlabel("Number of Flows")
# plt.show()

