"""
Environment2 .py
用于对比算法 获取omnet的delay，jitter 等值
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from scipy import stats
import subprocess
import networkx as nx
import pandas as pd
import json
import sys
import os
import time

from helper import pretty, softmax, MaxMinNormalization
from Traffic import Traffic
from K_shortest_path import k_shortest_paths
# from Reward_QoE import reward_QoE ,NN_training
from newrewardQoe import reward_QoE
from other_action import OtherAction
from helper import setup_exp, setup_run, parser, pretty, scale

OMTRAFFIC = 'Traffic.txt'
OMBALANCING = 'Balancing.txt'
OMROUTING = 'Routing.txt'
OMDELAY = 'Delay.txt'
# OMDBW = 'Bandwidth.txt'
OMPLR = 'PacketLossRate.txt'
OMJITTER = 'Jitter.txt'
ALLLOG = 'allLog.txt'
TRAFFICLOG = 'TrafficLog.csv'
BALANCINGLOG = 'BalancingLog.csv'
REWARDLOG = 'rewardLog.txt'
WHOLELOG = 'Log.csv'
OMLOG = 'omnetLog.csv'
BANDWIDTH = 'Bandwidth.txt'


# FROM MATRIX
def matrix_to_rl(matrix):
    '''return matrix 中不为 -1 的元素 array'''
    return matrix[(matrix!=-1)] 

matrix_to_log_v = matrix_to_rl

def matrix_to_omnet_v(matrix):
    '''return matrix的copy array'''
    return matrix.flatten()

def vector_to_file(vector, file_name, action):
    """将vector转换为string，写入file中"""
    string = ','.join(pretty(_) for _ in vector)
    with open(file_name, action) as file:
        return file.write(string + '\n')

# FROM FILE
def file_to_csv(file_name):
    '''reads file, outputs csv'''
    with open(file_name, 'r') as file:
        return file.readline().strip().strip(',')


def csv_to_vector(string, start, vector_num):
    v = np.asarray(string.split(',')[start:(start+vector_num)])
    M = v.astype(np.float)
    return M
    

def rl_reward(env, model, data_mean, data_std):
    '''根据env.env_D 和不同PRAEMIUM 计算reward '''
    delay = np.asarray(env.env_D)
    mask = delay == np.inf
    delay[mask] = len(delay)*np.max(delay[~mask])

    if env.PRAEMIUM == 'AVG':
        pass
        # reward = -np.mean(matrix_to_rl(delay))
        # reward = reward_QoE (env.env_Bw,env.env_D,env.env_J,env.env_L, model, data_mean, data_std)
    elif env.PRAEMIUM == 'MAX':
        reward = -np.max(matrix_to_rl(delay))
    elif env.PRAEMIUM == 'AXM':
        reward = -(np.mean(matrix_to_rl(delay)) + np.max(matrix_to_rl(delay)))/2
    elif env.PRAEMIUM == 'GEO':
        reward = -stats.gmean(matrix_to_rl(delay))
    elif env.PRAEMIUM == 'LOST':
        reward = -env.env_L
    return reward


# WRAPPER ITSELF
def omnet_wrapper(env):
    if env.ENV == 'label':
        sim = 'router'
    elif env.ENV == 'balancing':
        sim = 'balancer'

    prefix = ''
    if env.CLUSTER == 'arvei':
        prefix = '/scratch/nas/1/giorgio/rlnet/'

    simexe = prefix + 'omnet/' + sim + '/networkRL'
    simfolder = prefix + 'omnet/' + sim + '/'
    simini = prefix + 'omnet/' + sim + '/' + 'omnetpp.ini'

    try:
        # call omnet to calculate and get the output
        omnet_output = subprocess.check_output([simexe, '-n', simfolder, simini, env.folder + 'folder.ini']).decode()
    except Exception as e:
        omnet_output = e.stdout.decode()
        print(omnet_output)

    # vector_to_file([omnet_output],env.folder + 'omgoutput.csv', 'a')

    if 'Error' in omnet_output:
        omnet_output = omnet_output.replace(',', '')
        o_u_l = [_.strip() for _ in omnet_output.split('\n') if _ is not '']
        omnet_output = ','.join(o_u_l[4:])
    else:
        omnet_output = 'ok'

    vector_to_file([omnet_output], env.folder + OMLOG, 'a')


# label environment
class OmnetLinkweightEnv():

    def __init__(self, DDPG_config, folder):
        self.ENV = 'label'
        self.ROUTING = 'Linkweight'

        self.folder = folder

        self.ACTIVE_NODES = DDPG_config['ACTIVE_NODES']

        self.ACTUM = DDPG_config['ACTUM'] 
        # action:new or delete
        
        topology = 'omnet/router/NetworkAll.matrix'  
        # input the network matrix
        self.graph = nx.Graph(np.loadtxt(topology, dtype=int))
        if self.ACTIVE_NODES != self.graph.number_of_nodes():
            return False
        ports = 'omnet/router/NetworkAll.ports'
        self.ports = np.loadtxt(ports, dtype=int)
        self.flow_num = DDPG_config['FLOW_NUM']
        
        # self.a_dim = self.graph.number_of_edges() 
        self.a_dim  = self.flow_num*2
        # counts the total number of edges in the graph:

        # self.s_dim = self.ACTIVE_NODES**2 - self.ACTIVE_NODES    # traffic minus diagonal
        self.s_dim = self.flow_num*4
        self.STATUM = DDPG_config['STATUM']
        if self.STATUM == 'RT':
            self.s_dim *= 2    # traffic + routing table minus diagonals

        self.PRAEMIUM = DDPG_config['PRAEMIUM']

        capacity = self.ACTIVE_NODES * (self.ACTIVE_NODES -1)

        self.TRAFFIC = DDPG_config['TRAFFIC']
        
        # self.tgen = Traffic(self.ACTIVE_NODES, self.TRAFFIC, capacity, self.flow_num, '')

        self.CLUSTER = DDPG_config['CLUSTER'] if 'CLUSTER' in DDPG_config.keys() else False
        self.env_T = np.full((self.flow_num, 5), -1.0, dtype=int)
        # self.env_T = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # traffic
        self.env_W = np.full([self.a_dim], -1.0, dtype=float)           # weights
        self.env_R = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)    # routing
        self.env_Rn = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)   # routing (nodes)
        self.env_D = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # delay
        # TODO: env_D, env_L, env_J 的格式
        self.env_L = np.full(self.flow_num, -1.0, dtype=float) # lost
        self.env_J = np.full(self.flow_num, -1.0, dtype=float) # jitter
        self.env_Bw = np.full(self.flow_num, 10, dtype=float) # bandwidth of each flow， 假设flow数目为N
        self.env_S = np.full((self.flow_num, 4), -1.0, dtype=float) # state(bw, delay, jitter, loss)
        self.env_bw_original = np.full(self.flow_num, -1.0, dtype=float) 
        self.env_Path = []
        self.counter = 0
        self.env_flow_R = []
        self.env_choice = np.full(self.flow_num, 0, dtype=int) # 每条流的路径选择
        self.choice_id = 0
        self.altype = 0
        self.sort = ''

        self.chose_all_flow_path_set = [] # 保存所有流选择好的路径
        # return True


    def upd_env_T(self, matrix):
        self.env_T = np.asarray(matrix)
        # self.env_T[:,3] = vector
    def upd_env_T_tr(self, vector):
        self.env_T[:,3] = vector

    def upd_env_W(self, vector):
        self.env_W = np.asarray(softmax(vector))

    def list_to_file(self,file_name, action):
        """将list转换为string，写入file中"""
        # string = ','.join(pretty(_) for _ in path)
        file = open(file_name, action)
        for path in self.chose_all_flow_path_set:
            path = map(str, path)
            file.write(','.join(path))
            file.write('\n')

    def other_up(self,flow_num):
        self.flow_num = flow_num
        self.otheraction = OtherAction(self.altype,self.flow_num,self.choice_id, self.sort) # 第一组数据
        path_chose = self.otheraction.read_path_choice() # 得到选择的路径
        self.env_Path = path_chose
        # print('env_path', self.env_Path)
        self.upd_env_W(np.full([self.a_dim], 0.05, dtype=float)) 
        # self.env_W = self.otheraction.read_weight()
        # self.env_T = np.full((self.flow_num, 5), -1.0, dtype=int)
        self.env_T = self.otheraction.csv_traffic()
        print('tttt',self.env_T)
        print('ebdbd')
        
        self.env_T[:,3] = self.otheraction.read_band()

        # print( 'banddddd', self.env_T[:,0] )
        self.env_Bw = self.env_T[:,3]
        vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')
        
        self.env_flow_R = []
        for i in range(self.flow_num):
            # choice = self.env_choice[i]
            self.upd_env_R_from_k_flow(i)
            # self.env_flow_R.append(routing)
            # vector_to_file(matrix_to_omnet_v(routing), self.folder + 'Routing'+str(i)+'.txt', 'w')
        self.list_to_file(self.folder + 'flowpath.txt', 'w')
        # execute omnet
        omnet_wrapper(self)
        print('ok')

        """
        去掉omnet 假定数据进行测试
        """
        om_output_delay = file_to_csv(self.folder + OMDELAY)
        om_output_jitter = file_to_csv(self.folder + OMJITTER)
        om_output_plr = file_to_csv(self.folder + OMPLR)
        om_output_bandwidth = file_to_csv(self.folder + BANDWIDTH)


        self.upd_env_D(csv_to_vector(om_output_delay, 0, self.flow_num)) # 更新delay
        self.upd_env_L(csv_to_vector(om_output_plr, 0, self.flow_num)) #packt loss
        self.upd_env_J(csv_to_vector(om_output_jitter, 0, self.flow_num))
        self.env_Bw = csv_to_vector(om_output_bandwidth, 0, self.flow_num)

        # reward = reward_QoE(self.env_Bw, self.env_D,self.env_J,self.env_L, model, data_mean, data_std)
        reward = reward_QoE(self.flow_num, self.env_Bw, self.env_D, self.env_J, self.env_L, self.env_T)
        
        # log everything to file
        vector_to_file([reward], self.folder + REWARDLOG, 'a') # 将-reward 写入 rewardLog.csv

    def upd_env_R_from_k_flow(self, flow_id):
        self.upd_env_R()
        routing_ports = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)
        path = self.env_Path[flow_id]
        path_o_ports = []
        s = path[0]
        d = path[-1]
        for i in range(len(path)-1):
            r = path[i]
            next = path[i+1]
            port = self.ports[r][next]
            self.env_R[r][d] = port
            path_o_ports.append(r)
            path_o_ports.append(port)
        path_o_ports.append(path[-1])
        self.chose_all_flow_path_set.append(path_o_ports)
        # print('paht', path_o_ports)

        # self.env_R = np.asarray(routing_ports)
        # self.env_Rn = np.asarray(routing_nodes)
        t = np.asarray(self.env_R)
        return t

    def upd_env_R(self):
        weights = {}

        for e, w in zip(self.graph.edges(), self.env_W):
            weights[e] = w

        nx.set_edge_attributes(self.graph, 'weight', weights)

        routing_nodes = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)
        routing_ports = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)

        all_shortest = nx.all_pairs_dijkstra_path(self.graph)
        
        for s in range(self.ACTIVE_NODES):
            for d in range(self.ACTIVE_NODES):
                if s != d:
                    next = all_shortest[s][d][1]
                    port = self.ports[s][next]
                    routing_nodes[s][d] = next
                    routing_ports[s][d] = port
                else:
                    routing_nodes[s][d] = -1
                    routing_ports[s][d] = -1

        self.env_R = np.asarray(routing_ports)
        self.env_Rn = np.asarray(routing_nodes)

    # def upd_env_D(self, matrix):
    #     self.env_D = np.asarray(matrix)
    #     np.fill_diagonal(self.env_D, -1)
    def upd_env_D(self, vector):
        self.env_D = np.asarray(vector)

    def upd_env_L(self, vector):
        self.env_L = np.asarray(vector)

    def upd_env_J(self, vector):
        self.env_J = np.asarray(vector)

    def upd_env_Bw(self, vector):
        Df = self.env_T[:,4]
        df = self.env_bw_original
        dd = df*0.85
        new  =( df + (Df-df)*vector )
        V = new.copy()
        for i in range(self.flow_num):
            (n , D, d ) =(new[i], Df[i], df[i])
            V[i] = n.clip(min=d,max=D)

        self.env_Bw = np.asarray(V)
        

    def upd_env_S(self):
        self.env_S[:,0] = self.env_Bw # bandwidth
        self.env_S[:,1] = self.env_D # delay
        self.env_S[:,3] = self.env_J # jitter
        self.env_S[:,2] = self.env_L # loss
        
   
   
'''
mainfolder 代表 topo
new_action = inida35

'''
tag = sys.argv[1] # 标记次数
def omnetReward(mainfolder, sort):
    # mainfolder = "topo/india35/"
    # mainfolder ="sampler/"
    with open('DDPG.json') as jconfig:
        DDPG_config = json.load(jconfig)
        DDPG_config['EXPERIMENT'] = setup_exp()
    # model, data_mean, data_std = NN_training()
    # 5，10，15，20，25，30，35，40，45， 50 ,共10次， 
    # 放在每个不同的folder中,后缀表示算法的不同， 中间数字表示流的个数

    for i in range(10):
        DDPG_config['FLOW_NUM'] = (i+1)*5
        env = OmnetLinkweightEnv(DDPG_config, mainfolder)
        env.choice_id = i
        env.sort = sort
        env.altype = 0
        epoch = 't%.6f' % time.time()
        folder = mainfolder+epoch.replace('.', '')+'_'+str((i+1)*5)+"_tttt/"
        os.makedirs(folder, exist_ok=True)
        with open(folder + 'folder.ini', 'w') as ifile:
            ifile.write('[General]\n')
            ifile.write('**.folderName = "' + folder + '"\n')
            # flow_num = DDPG_config['FLOW_NUM']
            ifile.write('**.flow_num = ' + str(env.flow_num) + '\n')
        env.folder = folder
        env.other_up(env.flow_num)
        env.altype = 1
        folder = mainfolder+epoch.replace('.', '')+'_'+str((i+1)*5)+"_uuuu/"
        os.makedirs(folder, exist_ok=True)
        with open(folder + 'folder.ini', 'w') as ifile:
            ifile.write('[General]\n')
            ifile.write('**.folderName = "' + folder + '"\n')
            # flow_num = DDPG_config['FLOW_NUM']
            ifile.write('**.flow_num = ' + str(env.flow_num) + '\n')
        env.folder = folder
        env.other_up(env.flow_num)
        print('ss ok')


sort = 'Mat'+str(tag)
folder = sys.argv[2] + '/'
os.makedirs(folder, exist_ok=True)
omnetReward(folder, sort)