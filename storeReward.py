# -*- coding:utf-8 -*-

"""
绘制箱体图
Created on 2017.09.04 by ForestNeo
"""


import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# from collections import OrderedDict  
import os  ,sys
"""
根据产生的rewalog， 存reward
输入 runs storefilename两个变量
"""

      
def file_name(file_dir): 

    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        return dirs #当前路径下所有子目录  
        # print(files) #当前路径下所有非目录子文件
        # break
# df2 = pd.read_csv ("testfoo.csv" , encoding = "utf-8")
# print (df2)

def get_reward(folder):
    rewardlist = []
    filename = 'rewardLog.txt'
    df = pd.read_csv(folder +filename, header=None, sep=',')
    a = np.asarray(df) 
    for x in a:
        rewardlist.append(x[0])
    # print(rewardlist)
    return rewardlist


# Dataflst = {}
# datastore = dict()
# datastore = OrderedDict()
# for i in range(10):
#     filename = (i + 1) * 5
#     filename = str(filename)
#     datastore[filename] = []
# print(datastore)
rewardstorefolder = sys.argv[2] + '/'
dirname = sys.argv[1] + '/'

server_flow = file_name(dirname)
leng = int(len(server_flow)/10)
for i in range(len(server_flow)):
    ffile = server_flow[i]
    reward = get_reward(dirname+'/'+ffile+"/")
    
    avg = np.max(reward)
    # if ffile == 't1530021443995106_Mat1_flow_num_40':
        # print(reward)
        # print('avg',avg)
    start = ffile.find('m_') + 2
    end = len(ffile)
    flownum =ffile[start:end]
    print(flownum, avg)
    # print(flownum)
    #datastore[flownum].append(avg)
    if os.path.exists(rewardstorefolder) == False:
        os.makedirs(rewardstorefolder) 
    with open(rewardstorefolder+flownum+'.txt', 'w') as f:
        f.write(str(avg) +'\n')
# data = pd.DataFrame(datastore)

#draw
# data.boxplot()
# plt.ylabel("Mean MOS Value")
# plt.xlabel("Number of Flows")
# plt.show()

