from ddpg import main_play
from gates import genFlowsCSV
import pandas as pd
import numpy as np
# from Environment2 import omnetReward
import json
import os
import sys
# import OrderedDict
'''
循环跑 ddpg
'''

def get_reward(folder):
    # rewardlist = []
    filename = 'rewardLog.txt'
    df = pd.read_csv(folder +filename, header=None, sep=',')
    return np.asarray(df)
topo = 'india35'
folder = './topo/'+topo+"/"
with open('DDPG.json') as jconfig:
    DDPG_config = json.load(jconfig)
flow_num = DDPG_config['FLOW_NUM']
node_num = DDPG_config['ACTIVE_NODES']
cnt = DDPG_config['REPEATE']

# 生成新的traffic矩阵
allcntreward = []
folderall = []
# number = sys.argv[1]
cnt = 0
for k in range(cnt):
    rewardset = []
    folderset = []
    # print(number)
    genFlowsCSV(folder, node_num, 100, "test.csv") 
    flow_num = 30
    newfolder = main_play(flow_num)
