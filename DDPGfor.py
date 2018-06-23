from ddpg import main_play
from gates import genFlowsCSV
import pandas as pd
import numpy as np
# from Environment2 import omnetReward
import json
import os
import sys
import time
# import OrderedDict
'''
循环跑 ddpg
'''

def get_reward(folder):
    # rewardlist = []
    filename = 'rewardLog.txt'
    df = pd.read_csv(folder +filename, header=None, sep=',')
    return np.asarray(df)
# topo = 'india35'
topo = 'germany50'
folder = './topo/'+topo+"/"
with open('DDPG.json') as jconfig:
    DDPG_config = json.load(jconfig)
flow_num = DDPG_config['FLOW_NUM']
node_num = DDPG_config['ACTIVE_NODES']
cnt = DDPG_config['REPEATE']

# 生成新的traffic矩阵
allcntreward = []
folderall = []
number = sys.argv[1]
for k in range(cnt):
    rewardset = []
    folderset = []
    number = k
    epoch = 't%.6f' % time.time()
    # epoch = 'test'
    # flows_india35_1.csv =csvname
    flowfoler = 'store_flows/'
    flowfoler += epoch.replace('.', '') + "_flows_"+topo+"_"+str(k)+str(number)+".csv"
    genFlowsCSV(folder, node_num, 100, flowfoler)   # flows_india35_1.csv
    for i in range(10):
        # 从flow=5开始，跑到flow = 50
        flow_num = (i + 1) * 5
        newfolder = main_play(flow_num, number, flowfoler) # training
        rewardlist = get_reward(newfolder)
        avg_reward = np.mean(rewardlist)
        rewardset.append(avg_reward)
        folderset.append(newfolder)
        print("step -----" + str(i))
        # createfolder = folder + "ut_" + str(i) + '/'
        # os.makedirs(createfolder, exist_ok=True)
        # omnetReward(createfolder) # 其他不同算法
    # print(rewardset)
    # print(folderset)
    allcntreward.append(rewardlist)
    folderall.append(folderset)
# print(allcntreward)
print(folderall)



