"""
Environment.py
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from scipy import stats
import subprocess
import networkx as nx
import pandas as pd
import sklearn.preprocessing
from helper import pretty, softmax, MaxMinNormalization
from keras.utils import np_utils

from Traffic import Traffic
from K_shortest_path import k_shortest_paths
from Reward_QoE import reward_QoE ,NN_training,testmodel
# from newrewardQoe import reward_QoE
import time
from genLinkBand import getlinkuiti
OMTRAFFIC = 'Traffic.txt'
OMBALANCING = 'Balancing.txt'
OMROUTING = 'Routing.txt'
OMDELAY = 'Delay.txt'
# OMDBW = 'Bandwidth.txt'
OMPLR = 'PacketLossRate.txt'
OMJITTER = 'Jitter.txt'
BANDWIDTH = 'Bandwidth.txt'
ALLLOG = 'allLog.txt'
TRAFFICLOG = 'TrafficLog.csv'
BALANCINGLOG = 'BalancingLog.csv'
REWARDLOG = 'rewardLog.txt'
WHOLELOG = 'Log.csv'
OMLOG = 'omnetLog.csv'


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

def string_to_file(string, file_name, action):
    with open(file_name, action) as file:
        return file.write(string +'\n')

# FROM FILE
def file_to_csv(file_name):
    '''reads file, outputs csv'''
    with open(file_name, 'r') as file:
        return file.readline().strip().strip(',')

def csv_to_matrix(string, nodes_num):
    '''reads text, outputs matrix'''
    v = np.asarray(tuple(float(x) for x in string.split(',')[:nodes_num**2]))
    M = np.split(v, nodes_num)
    return np.vstack(M)

def csv_to_vector(string, start, vector_num):
    v = np.asarray(string.split(',')[start:(start+vector_num)])
    M = v.astype(np.float)
    return M
    

def csv_to_lost(string):
    return float(string.split(',')[-1])

def paths_to_string(paths):
    string = ''
    for i in paths:
        for j in i:
            string += ','.join(str(e) for e in j )
            string +='\n'
    return string


# FROM RL
def rl_to_matrix(vector, nodes_num):
    '''vector to matrix'''
    M = np.split(vector, nodes_num)
    for _ in range(nodes_num):
        M[_] = np.insert(M[_], _, -1)
    return np.vstack(M)


# TO RL
def rl_state(env):
    '''return env_S'''
    if env.STATUM == 'RT':
        return np.concatenate((matrix_to_rl(env.env_B), matrix_to_rl(env.env_T)))
    elif env.STATUM == 'T':
        # return matrix_to_rl(env.env_T)
        # env.upd_env_S()
        # return matrix_to_rl(env.env_S)
       
        # tm = softmax(env.env_Bw)
        # tm = matrix_to_rl(tm)
        temp = np.concatenate((
            (softmax(env.env_Bw)), \
            (MaxMinNormalization(env.env_D,'d')),\
            (MaxMinNormalization(env.env_J,'j')),\
            (env.env_L)\
            
        ))
        print(temp.shape[0])
        # print(temp.shape[1])
        # print(temp)
        temp =  np.asarray((temp))
        # tm = tm.reshape(env.flow_num, 1)
        # size = tm.shape[0]
        # print(tm)
        # print(tm.shape[1])
        # print()
       
        return temp
        # temp = np.asarray(MaxMinNormalization(matrix_to_rl(env.env_Bw)))
        # temp = np.concatenate((
        #     MaxMinNormalization(matrix_to_rl(env.env_Bw)), \
        #     # MaxMinNormalization(matrix_to_rl(env.env_D)), \
        #     # MaxMinNormalization(matrix_to_rl(env.env_J)), \
        #     # MaxMinNormalization(matrix_to_rl(env.env_L))\
        #     ))
        # temp = np.concatenate((
        #     softmax(matrix_to_rl(env.env_Bw)), \
        #     softmax(matrix_to_rl(env.env_D)), \
        #     softmax(matrix_to_rl(env.env_J)), \
        #     softmax(matrix_to_rl(env.env_L))\
        #     ))
        # return temp

def rl_reward(env, model, data_mean, data_std):
   pass
       


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

        vector_to_file([omnet_output],env.folder + 'omgoutput.csv', 'a')

    if 'Error' in omnet_output:
        omnet_output = omnet_output.replace(',', '')
        o_u_l = [_.strip() for _ in omnet_output.split('\n') if _ is not '']
        omnet_output = ','.join(o_u_l[4:])
        # vector_to_file([omnet_output], env.folder + OMLOG, 'a')

    else:
        omnet_output = 'ok'



def ned_to_capacity(env):
    if env.ENV == 'label':
        sim = 'router'
    elif env.ENV == 'balancing':
        sim = 'balancer'
    NED = 'omnet/' + sim + '/NetworkAll.ned'

    capacity = 0

    with open(NED) as nedfile:
        for line in nedfile:
            if "SlowChannel" in line and "<-->" in line:
                capacity += 3
            elif "MediumChannel" in line and "<-->" in line:
                capacity += 5
            elif "FastChannel" in line and "<-->" in line:
                capacity += 10
            elif "Channel" in line and "<-->" in line:
                capacity += 10

    return capacity or None


# label environment
class OmnetLinkweightEnv():

    def __init__(self, DDPG_config, folder, flowfile):
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
            print('nodes num 不符合图')
            return 
        ports = 'omnet/router/NetworkAll.ports'
        self.ports = np.loadtxt(ports, dtype=int)
        self.flow_num = DDPG_config['FLOW_NUM']
        # self.a_dim = self.graph.number_of_edges() 
        self.a_dim  = self.flow_num*4
        # counts the total number of edges in the graph:

        # self.s_dim = self.ACTIVE_NODES**2 - self.ACTIVE_NODES    # traffic minus diagonal
        self.s_dim = self.flow_num*4 # 输入包括带宽,delay,jitter,loss
        self.STATUM = DDPG_config['STATUM']
        if self.STATUM == 'RT':
            self.s_dim *= 2    # traffic + routing table minus diagonals

        self.PRAEMIUM = DDPG_config['PRAEMIUM']

        capacity = self.ACTIVE_NODES * (self.ACTIVE_NODES -1)

        self.TRAFFIC = DDPG_config['TRAFFIC']
        
        self.tgen = Traffic(self.ACTIVE_NODES, self.TRAFFIC, capacity, self.flow_num, flowfile)
        self.path_file = flowfile.replace('flows','path')
        self.path_choose_file = flowfile.replace('flows','path_chosen_thr')
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
        self.reward = 0
        self.env_choice = np.full(self.flow_num, 0, dtype=int) # 每条流的路径选择
        self.env_path_set = [] # 存储选择后的所有流的path
        self.chose_all_flow_path_set = [] # 保存所有流选择好的路径

        self.model = None
        self.data_mean = None
        self.data_std = None
        # return True
    
    def upd_env_T(self, matrix):
        self.env_T = np.asarray(matrix)
        # self.env_T[:,3] = vector
    def upd_env_T_tr(self, vector):
        self.env_T[:,3] = vector

    def upd_env_W(self, vector):
        self.env_W = np.asarray(softmax(vector))

    def read_path(self):
        all_path = pd.read_csv( self.path_file , header=None, sep=',')
        index = 0
        self.env_Path = []
        while index < len(all_path):
            flow_path = []
            for i in range(3):
                path = all_path.iloc[index + i,:]
                path = np.asarray(path)-1
                flow_path.append(path[path!=-1])
            index += 3
            self.env_Path.append(flow_path)
    

    def read_path_choice(self):
        choice  = pd.read_csv(self.path_choose_file, header=None, sep=',')# 第id行数据
        result = []
        id = int(self.flow_num/5)-1
        for i in range(self.flow_num):
            
            path_id = choice.iloc[id][i]-1 # indix 从0开始
            path_id = (path_id - i*3 )% 3
            print(path_id)
            result.append(path_id)

           
        return result

    def set_env_R_K(self,k=3):
        '''拿到每条流得到的path集合'''
        self.read_path()
        # sour_des = self.env_T[:,1:3]
        # all_path = []
        # for x in sour_des:
        #     s = int(x[0])
        #     d = int(x[1])
        #     (length, path) = k_shortest_paths(self.graph.copy(), s, d, k)
        #     all_path.append(path)
        # self.env_Path = all_path
        
       


    def upd_env_R_from_k_flow(self, flow_id,choice):
        '''
        根据每条流的路径，结合ports矩阵，得到节点下一跳的出口port矩阵
        '''
        self.upd_env_R()
        routing_ports = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)
        path = self.env_Path[flow_id][choice]
        self.env_path_set.append(path)
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

    def upd_env_R_from_R(self, routing):
        routing_nodes = np.fromstring(routing, sep=',', dtype=int)
        M = np.split(np.asarray(routing_nodes), self.ACTIVE_NODES)
        routing_nodes = np.vstack(M)

        routing_ports = np.zeros([self.ACTIVE_NODES]*2, dtype=int)

        for s in range(self.ACTIVE_NODES):
            for d in range(self.ACTIVE_NODES):
                if s != d:
                    next = routing_nodes[s][d]
                    port = self.ports[s][next]
                    routing_ports[s][d] = port
                else:
                    routing_ports[s][d] = -1

        self.env_R = np.asarray(routing_ports)
        self.env_Rn = np.asarray(routing_nodes)

    # def upd_env_D(self, matrix):
    #     self.env_D = np.asarray(matrix)
    #     np.fill_diagonal(self.env_D, -1)
    def upd_env_D(self, vector):
        self.env_D = np.asarray(vector)
        print('avg-delay=',self.env_D.mean())

    def upd_env_L(self, vector):
        self.env_L = np.asarray(vector)
        print('avg-loss=',self.env_L.mean())

    def upd_env_J(self, vector):
        self.env_J = np.asarray(vector)
        print('avg-jitter=',self.env_J.mean())

    def upd_env_Bw(self, vector):
        Df = self.env_T[:,4]
        df = self.env_bw_original
        dd = df
        new  =( dd + (Df-df)*vector )
        # print(new-dd)

        V = new.copy()
        for i in range(self.flow_num):
            (n , D, d ) =(new[i], Df[i], df[i])
            V[i] = n.clip(min=d,max=D)

        self.env_Bw = np.asarray(V)
        # getlinkuiti(pathset,bw,flow_num,node):
        getlinkuiti(self.env_path_set, self.env_Bw, self.flow_num,  self.ACTIVE_NODES)
        print('avg-band=', self.env_Bw.mean())

    def upd_env_S(self):
        self.env_S[:,0] = self.env_Bw # bandwidth
        self.env_S[:,1] = self.env_D # delay
        self.env_S[:,3] = self.env_J # jitter
        self.env_S[:,2] = self.env_L # loss
    

    def upd_env_choice(self, vector):
        # 根据 action 
        choice = np.full(self.flow_num, 0, dtype=int)
        for i in range(self.flow_num):
            path = list(vector[i*3 : (i+1)*3])
            # print(path)
            # print(max(path))
            choice[i] = path.index(max(path))
            


        # input()
        self.env_choice = np.asarray(choice)
            

    def log_everything(self):
        head = ['flow_id', 's', 'd', 'bandwidth','delay','jitter','packetloss']
        df =  pd.DataFrame (columns = head)
        for i in range(self.flow_num):
            flow_id = i
            s = self.env_T[i][1]
            des = self.env_T[i][2]
            b = self.env_T[i][3]
            delay = self.env_D[i]
            j = self.env_J[i]
            p = self.env_L[i]
            l = [flow_id, s, des, b, delay,j,p]
            df.loc[i] = l
        with open(self.folder + ALLLOG, 'a') as f:
            df.to_csv(f, header=False)
        

    def my_log_header(self):
        head = ['flow_id', 's', 'd', 'bandwidth','delay','jitter','packetloss']
        # df.loc[0] = np.arange(0,7)
        df = pd.DataFrame (columns = head)
        df.to_csv (self.folder + ALLLOG , encoding = 'utf-8')


    def logheader(self, easy=False):
        nice_matrix = np.chararray([self.flow_num]*2, itemsize=20)
        for i in range(self.flow_num):
            for j in range(self.flow_num):
                nice_matrix[i][j] = str(i) + '-' + str(j)
        np.fill_diagonal(nice_matrix, '_')
        nice_list = list(nice_matrix[(nice_matrix!=b'_')])

        tra_matrix = np.chararray((self.flow_num, 5), itemsize=20)
        for i in range(self.flow_num):
            for j in range(5):
                tra_matrix[i][j] = str(i) + '-' + str(j)
        np.fill_diagonal(tra_matrix, '_')
        tra_list = list(tra_matrix[(tra_matrix!=b'_')])
        
        rnw_matrix = np.chararray([self.ACTIVE_NODES]*2, itemsize=20)
        for i in range(self.ACTIVE_NODES):
            for j in range(self.ACTIVE_NODES):
                rnw_matrix[i][j] = str(i) + '-' + str(j)
        np.fill_diagonal(rnw_matrix, '_')
        r_list = list(rnw_matrix[(rnw_matrix!=b'_')])


        th = ['t' + _.decode('ascii') for _ in tra_list]
        bh = ['b' + _.decode('ascii') for _ in nice_list]
        dh = ['d' + _.decode('ascii') for _ in nice_list]
        jh = ['j' + _.decode('ascii') for _ in nice_list]
        lh = ['l' + _.decode('ascii') for _ in nice_list]
        rnh = ['r' + _.decode('ascii') for _ in r_list]
        ah = ['a' + str(_[0]) + '-' + str(_[1]) for _ in self.graph.edges()]
        header = ['counter'] + th + bh + dh + jh + lh + ['reward'] + ah + rnh
        if easy:
            header = ['counter', 'lost', 'AVG', 'MAX', 'AXM', 'GEO']
        vector_to_file(header, self.folder + WHOLELOG, 'w')

    
    def render(self):
        return

    def list_to_file(self,file_name, action):
        """将list转换为string，写入file中"""
        # string = ','.join(pretty(_) for _ in path)
        file = open(file_name, action)
        for path in self.chose_all_flow_path_set:
            path = map(str, path)
            file.write(','.join(path))
            file.write('\n')
            
    def read_band(self):
        band = pd.read_csv('data_new_multi/Mat1/bw_thr_Mat1.csv', header=None, sep=',')
        index_id = int(self.flow_num / 5 - 1 )
        t = band.iloc[index_id]
        t = np.asarray(t[:self.flow_num])
        # print(t)
        return t
   

    def reset(self, easy=False,):
        
        if self.counter != 0:
            return None
        # self.logheader()
        # routing
        self.model, self.data_mean, self.data_std = testmodel()
        self.upd_env_W(np.full([self.a_dim], 0.05, dtype=float)) 
        # if self.ACTUM == 'DELTA':
        #     vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
            # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")
        
        # traffic
        self.env_T = np.asarray(self.tgen.generate())

        self.env_bw_original = self.env_T[:,3].copy()
        self.env_Bw = self.read_band().copy()
        self.env_T[:,3] = self.env_Bw

        vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')
        self.set_env_R_K(3)

        # self.env_chocie = np.random.randint(low=0,high=4,size=1)
        self.env_choice = self.read_path_choice()
        self.env_flow_R = []
        tforR1 = time.time()
        self.chose_all_flow_path_set = []
        self.env_path_set = []
        # self.env_chocie = [0,0,0,0,1]
        for i in range(self.flow_num):
            choice = self.env_choice[i]
            routing = self.upd_env_R_from_k_flow(i,choice)
            self.env_flow_R.append(routing)
            #vector_to_file(matrix_to_omnet_v(routing), self.folder + 'Routing'+str(i)+'.txt', 'w')

        self.list_to_file(self.folder + 'flowpath.txt', 'w')
        # print("Routing time = " + str(time.time()- tforR1))
        t1 = time.time()
        # execute omnet
        omnet_wrapper(self)

        t2 = time.time()
        # print("one omnet spend time = " + str(t2-t1))
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
        self.my_log_header()
        self.log_everything()
        self.reward = reward_QoE(self.env_Bw, self.env_D, self.env_J, self.env_L, self.model, self.data_mean, self.data_std)
        # self.reward = reward_QoE(self.flow_num, self.env_Bw, self.env_D, self.env_J, self.env_L, self.env_T)
        print('reset-reward',self.reward)

        # 前面的都作废
        self.env_Bw = self.env_T[:,4]
        # self.env_Bw = self.genRandomBand()
        # self.env_T[:,3] = self.env_Bw

        return rl_state(self) # if STATUM==T, return self.env_S 
    
    def genRandomBand(self):
        a = np.random.rand(1, self.flow_num)
        b = (self.env_T[:,4] - self.env_T[:,3]).copy()
        df = self.env_T[:,3].copy()
        band = b * a + df
        return np.asarray(band)

        

    def test(self,model, data_mean, data_std):
        while True:
            b = input('put band :')
            if b == 'quit':
                break
            b = float(b)
            band =[]
            band.append(b)

            b = input('put delay :')
            if b == 'quit':
                break
            b = float(b)
            delay = []
            delay.append(b)

            b = input('put jitter :')
            if b == 'quit':
                break
            b = float(b)
            jitter = []
            jitter.append(b)

            b = input('put loss :')
            if b == 'quit':
                break
            b = float(b)
            loss = []
            loss.append(b)


            jitter = np.asarray(jitter)
            loss = np.asarray(loss)
            delay = np.asarray(delay)
            band = np.asarray(band)
            # reward = reward_QoE (band,delay,jitter,loss, model, data_mean, data_std)
            # print(reward)

    def step(self, action):
        '''
        根据action得到 计算得到相应的带宽和路径，进入omnet中验证，得到一个新的reward值和state(放入随机的bandwidth)
        '''
        self.counter += 1
        # print('action',action)
        # self.upd_env_W(action[:self.graph.number_of_edges()]) # 特殊函数处理过后的概率数字作为weights
        bd = action[:self.flow_num]
        # print('bd',bd)
        # self.env_bw_original = bd
        self.upd_env_Bw(bd) # update Bandwidth
        self.upd_env_T_tr(self.env_Bw)
        

        vector_to_file(matrix_to_log_v(self.env_Bw), self.folder + 'bandwidth_store.txt','a')
        self.upd_env_choice(action[self.flow_num:])
        # self.env_choice = [0,0,0,0,1]
        print('choice :', self.env_choice)

        self.env_flow_R = []
        tforR1 = time.time()
        self.chose_all_flow_path_set = []
        self.env_path_set = []
        for i in range(self.flow_num):
            choice = self.env_choice[i]
            routing = self.upd_env_R_from_k_flow(i,choice)
            self.env_flow_R.append(routing)
            #vector_to_file(matrix_to_omnet_v(routing), self.folder + 'Routing'+str(i)+'.txt', 'w')
        self.list_to_file(self.folder + 'flowpath.txt', 'w')
        # print("Routing time = " + str(time.time()- tforR1))
         # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        # self.upd_env_R_K(3)

        # write to file input for Omnet: Routing - path
        # string_to_file(paths_to_string(self.env_Path), self.folder + OMROUTING, 'w')
        # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # execute omnet
        t1 = time.time()
        omnet_wrapper(self)
        # om_output = '3.57671,3.425,3.58462,6.41108,7.72395,6.63175,7.27326,4.81703,11.0406,7.998,8.3895,8.75981,8.43494,0.310579,2.78788'
        t2 = time.time()
        # print("omnet-spend time = " + str(t2-t1))
        # TODO: 读取omnet输出的delay，loss，jitter，loss信息

        # read Omnet's output: Delay and Lost
        om_output_delay = file_to_csv(self.folder + OMDELAY)
        om_output_jitter = file_to_csv(self.folder + OMJITTER)
        om_output_plr = file_to_csv(self.folder + OMPLR)
        om_output_bandwidth = file_to_csv(self.folder + BANDWIDTH)

        self.upd_env_D(csv_to_vector(om_output_delay, 0, self.flow_num)) # 更新delay
        self.upd_env_L(csv_to_vector(om_output_plr, 0, self.flow_num)) #packt loss
        self.upd_env_J(csv_to_vector(om_output_jitter, 0, self.flow_num))
        self.env_Bw = csv_to_vector(om_output_bandwidth, 0, self.flow_num)
       
        # self.upd_env_S()
        
        reward = reward_QoE(self.env_Bw, self.env_D, self.env_J, self.env_L, self.model, self.data_mean, self.data_std)
        # reward = reward_QoE(self.flow_num, self.env_Bw, self.env_D, self.env_J, self.env_L, self.env_T)
        # log everything to file
        vector_to_file([reward], self.folder + REWARDLOG, 'a') # 将-reward 写入 rewardLog.csv
        vector_to_file(matrix_to_log_v(self.env_D), self.folder + 'Dealy_store.txt','a')
        # vector_to_file(matrix_to_log_v(self.env_W), self.folder + 'weights.csv','a')
        vector_to_file(matrix_to_log_v(self.env_J), self.folder + 'Jitter_store.txt','a')
        vector_to_file(matrix_to_log_v(self.env_L), self.folder + 'loss_store.txt','a')
        # 生成下一组的带宽值

        # self.env_Bw = self.genRandomBand()
        # self.env_T[:,3] = self.env_Bw.copy()
        
        new_state = rl_state(self)   # 根据bandwith来得到新的state

        # self.log_everything()
        # return new status and reward
        # reward * 100 突出分数
         
        
        self.reward = reward
       

        # self.test( self.model, self.data_mean, self.data_std)
        return new_state, reward, 0

    
   
    def end(self):
        return
