"""
Environment.py
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from scipy import stats
import subprocess
import networkx as nx
import pandas as pd

from helper import pretty, softmax, MaxMinNormalization
from Traffic import Traffic
from K_shortest_path import k_shortest_paths
from Reward_QoE import reward_QoE 


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
        env.upd_env_S()
        # return matrix_to_rl(env.env_S)
        temp = np.concatenate((
            MaxMinNormalization(matrix_to_rl(env.env_Bw)), \
            MaxMinNormalization(matrix_to_rl(env.env_D)), \
            MaxMinNormalization(matrix_to_rl(env.env_J)), \
            MaxMinNormalization(matrix_to_rl(env.env_L))\
            ))
        return temp

def rl_reward(env, model, data_mean, data_std):
    '''根据env.env_D 和不同PRAEMIUM 计算reward '''
    delay = np.asarray(env.env_D)
    mask = delay == np.inf
    delay[mask] = len(delay)*np.max(delay[~mask])

    if env.PRAEMIUM == 'AVG':
        # reward = -np.mean(matrix_to_rl(delay))
        reward = reward_QoE (env.env_Bw,env.env_D,env.env_J,env.env_L, model, data_mean, data_std)
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

    # vector_to_file([omnet_output],env.folder + 'omgoutput.csv', 'a')

    if 'Error' in omnet_output:
        omnet_output = omnet_output.replace(',', '')
        o_u_l = [_.strip() for _ in omnet_output.split('\n') if _ is not '']
        omnet_output = ','.join(o_u_l[4:])
    else:
        omnet_output = 'ok'

    vector_to_file([omnet_output], env.folder + OMLOG, 'a')


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


# balancing environment
class OmnetBalancerEnv():

    def __init__(self, DDPG_config, folder):
        self.ENV = 'balancing'
        self.ROUTING = 'Balancer'

        self.folder = folder

        self.ACTIVE_NODES = DDPG_config['ACTIVE_NODES']

        self.ACTUM = DDPG_config['ACTUM']
        self.a_dim = self.ACTIVE_NODES**2 - self.ACTIVE_NODES     # routing table minus diagonal

        self.s_dim = self.ACTIVE_NODES**2 - self.ACTIVE_NODES    # traffic minus diagonal

        self.STATUM = DDPG_config['STATUM']
        if self.STATUM == 'RT':
            self.s_dim *= 2    # traffic + routing table minus diagonals

        if 'MAX_DELTA' in DDPG_config.keys():
            self.MAX_DELTA = DDPG_config['MAX_DELTA']

        self.PRAEMIUM = DDPG_config['PRAEMIUM']

        capacity = self.ACTIVE_NODES * (self.ACTIVE_NODES -1)

        self.TRAFFIC = DDPG_config['TRAFFIC']
        self.tgen = Traffic(self.ACTIVE_NODES, self.TRAFFIC, capacity)

        self.CLUSTER = DDPG_config['CLUSTER'] if 'CLUSTER' in DDPG_config.keys() else False

        self.env_T = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # traffic
        self.env_B = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # balancing
        self.env_D = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # delay
        self.env_L = -1.0  # lost packets

        self.counter = 0


    def upd_env_T(self, matrix):
        '''update self.env_T and fill_digonal'''
        self.env_T = np.asarray(matrix)
        np.fill_diagonal(self.env_T, -1)

    def upd_env_B(self, matrix):
        '''return env_B fill diagonal with -1 '''
        self.env_B = np.asarray(matrix)
        np.fill_diagonal(self.env_B, -1)

    def upd_env_D(self, matrix):
        '''update the value of self.env_D, and fill_digonal'''
        self.env_D = np.asarray(matrix)
        np.fill_diagonal(self.env_D, -1)

    def upd_env_L(self, number):
        '''update self.env_L = number'''
        self.env_L = number


    def logheader(self):
        nice_matrix = np.chararray([self.ACTIVE_NODES]*2, itemsize=20)
        for i in range(self.ACTIVE_NODES):
            for j in range(self.ACTIVE_NODES):
                nice_matrix[i][j] = str(i) + '-' + str(j)
        np.fill_diagonal(nice_matrix, '_')
        nice_list = list(nice_matrix[(nice_matrix!=b'_')])
        th = ['t' + _.decode('ascii') for _ in nice_list]
        rh = ['r' + _.decode('ascii') for _ in nice_list]
        dh = ['d' + _.decode('ascii') for _ in nice_list]
        if self.STATUM == 'T':
            sh = ['s' + _.decode('ascii') for _ in nice_list]
        elif self.STATUM == 'RT':
            sh = ['sr' + _.decode('ascii') for _ in nice_list] + ['st' + _.decode('ascii') for _ in nice_list]
        ah = ['a' + _.decode('ascii') for _ in nice_list]
        header = ['counter'] + th + rh + dh + ['lost'] + sh + ah + ['reward']
        vector_to_file(header, self.folder + WHOLELOG, 'w')


    def render(self):
        return True


    def reset(self):
        if self.counter != 0:
            return None

        self.logheader()

        # balancing
        self.upd_env_B(np.full([self.ACTIVE_NODES]*2, 0.50, dtype=float))
        if self.ACTUM == 'DELTA':
            vector_to_file(matrix_to_omnet_v(self.env_B), self.folder + OMBALANCING, 'w')

        # traffic
        self.upd_env_T(self.tgen.generate())

        vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        return rl_state(self)


    def step(self, action):
        self.counter += 1

        # define action: NEW or DELTA
        if self.ACTUM == 'NEW':
            # bound the action
            self.upd_env_B(rl_to_matrix(np.clip(action, 0, 1), self.ACTIVE_NODES))
        if self.ACTUM == 'DELTA':
            # bound the action
            self.upd_env_B(rl_to_matrix(np.clip(action * self.MAX_DELTA + matrix_to_rl(self.env_B), 0, 1), self.ACTIVE_NODES))

        # write to file input for Omnet: Balancing
        vector_to_file(matrix_to_omnet_v(self.env_B), self.folder + OMBALANCING, 'w')

        # execute omnet
        omnet_wrapper(self)

        # read Omnet's output: Delay and Lost
        om_output = file_to_csv(self.folder + OMDELAY)
        self.upd_env_D(csv_to_matrix(om_output, self.ACTIVE_NODES))
        self.upd_env_L(csv_to_lost(om_output))

        reward = rl_reward(self)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARDLOG, 'a')
        cur_state = rl_state(self)
        log = np.concatenate(([self.counter], matrix_to_log_v(self.env_T), matrix_to_log_v(self.env_B), matrix_to_log_v(self.env_D), [self.env_L], cur_state, action, [-reward]))
        vector_to_file(log, self.folder + WHOLELOG, 'a')

        # generate traffic for next iteration
        self.upd_env_T(self.tgen.generate())
        # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        new_state = rl_state(self)
        # return new status and reward
        return new_state, reward, 0


    def end(self):
        return


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
        
        self.tgen = Traffic(self.ACTIVE_NODES, self.TRAFFIC, capacity, self.flow_num)

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
    
    def upd_env_T(self, matrix):
        self.env_T = np.asarray(matrix)
        # self.env_T[:,3] = vector
    def upd_env_T_tr(self, vector):
        self.env_T[:,3] = vector

    def upd_env_W(self, vector):
        self.env_W = np.asarray(softmax(vector))

    def set_env_R_K(self,k=3):
        sour_des = self.env_T[:,1:3]
        all_path = []
        for x in sour_des:
            s = int(x[0])
            d = int(x[1])
            (length, path) = k_shortest_paths(self.graph.copy(), s, d, k)
            all_path.append(path)
        self.env_Path = all_path
       

    def upd_env_R_from_k_flow(self, flow_id,choice):
        self.upd_env_R()
        routing_ports = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)
        path = self.env_Path[flow_id][choice]
        s = path[0]
        d = path[-1]
        for i in range(len(path)-1):
            r = path[i]
            next = path[i+1]
            port = self.ports[r][next]
            self.env_R[r][d] = port

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
        
    def upd_env_choice(self, vector):
        choice = np.full(self.flow_num, 0, dtype=int)
        for i in range(self.flow_num):
            x = vector[i]
            if x >= 0 and x < (0.1/3):
                choice[i] = 0
            elif x >= (0.1/3) and x < (0.2/3):
                choice[i] = 1
            else: 
                choice[i] = 2
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

    def reset(self, easy=False):
        if self.counter != 0:
            return None
        # self.logheader()
        # routing
        self.upd_env_W(np.full([self.a_dim], 0.05, dtype=float)) 
        # if self.ACTUM == 'DELTA':
        #     vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
            # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")
        
        # traffic
        self.env_T = np.asarray(self.tgen.generate())
        self.env_Bw = self.env_T[:,3]
        self.env_bw_original = self.env_Bw
        vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')
        self.set_env_R_K(3)

        self.env_chocie = np.random.randint(low=0,high=4,size=1)
        
        self.env_flow_R = []
        for i in range(self.flow_num):
            choice = self.env_choice[i]
            routing = self.upd_env_R_from_k_flow(i,choice)
            self.env_flow_R.append(routing)
            vector_to_file(matrix_to_omnet_v(routing), self.folder + 'Routing'+str(i)+'.txt', 'w')

        # execute omnet
        omnet_wrapper(self)
        """
        去掉omnet 假定数据进行测试
        """
        om_output_delay = file_to_csv(self.folder + OMDELAY)
        om_output_jitter = file_to_csv(self.folder + OMJITTER)
        om_output_plr = file_to_csv(self.folder + OMPLR)

        self.upd_env_D(csv_to_vector(om_output_delay, 0, self.flow_num)) # 更新delay
        self.upd_env_L(csv_to_vector(om_output_plr, 0, self.flow_num)) #packt loss
        self.upd_env_J(csv_to_vector(om_output_jitter, 0, self.flow_num))
        self.my_log_header()
        self.log_everything()

        return rl_state(self) # if STATUM==T, return self.env_S 
    
    


    def step(self, action, model, data_mean, data_std):
        self.counter += 1
        
        # self.upd_env_W(action[:self.graph.number_of_edges()]) # 特殊函数处理过后的概率数字作为weights

        
        bd = action[self.flow_num:]
        self.env_bw_original = bd
        self.upd_env_Bw(bd) # update Bandwidth

        self.upd_env_T_tr(self.env_Bw)
        self.upd_env_choice(action[:self.flow_num])
        self.env_flow_R = []
        for i in range(self.flow_num):
            choice = self.env_choice[i]
            routing = self.upd_env_R_from_k_flow(i,choice)
            self.env_flow_R.append(routing)
            
            vector_to_file(matrix_to_omnet_v(routing), self.folder + 'Routing'+str(i)+'.txt', 'w')
         # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        # self.upd_env_R_K(3)

        # write to file input for Omnet: Routing - path
        # string_to_file(paths_to_string(self.env_Path), self.folder + OMROUTING, 'w')
        # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # execute omnet

        omnet_wrapper(self)
        # om_output = '3.57671,3.425,3.58462,6.41108,7.72395,6.63175,7.27326,4.81703,11.0406,7.998,8.3895,8.75981,8.43494,0.310579,2.78788'
        
        # TODO: 读取omnet输出的delay，loss，jitter，loss信息

        # read Omnet's output: Delay and Lost
        om_output_delay = file_to_csv(self.folder + OMDELAY)
        om_output_jitter = file_to_csv(self.folder + OMJITTER)
        om_output_plr = file_to_csv(self.folder + OMPLR)

        self.upd_env_D(csv_to_vector(om_output_delay, 0, self.flow_num)) # 更新delay
        self.upd_env_L(csv_to_vector(om_output_plr, 0, self.flow_num)) #packt loss
        self.upd_env_J(csv_to_vector(om_output_jitter, 0, self.flow_num))
       
        # self.upd_env_S()

        reward = rl_reward(self, model, data_mean, data_std)
        # log everything to file
        vector_to_file([reward], self.folder + REWARDLOG, 'a') # 将-reward 写入 rewardLog.csv
        # cur_state = rl_state(self) # env_S

        # log = np.concatenate(([self.counter], matrix_to_log_v(self.env_T), \
        # matrix_to_log_v(self.env_Bw), matrix_to_log_v(self.env_D), matrix_to_log_v(self.env_J), \
        # matrix_to_log_v(self.env_L),\
        # [-reward])
        # vector_to_file(log, self.folder + WHOLELOG, 'a')
        vector_to_file(matrix_to_log_v(self.env_D), self.folder + 'Dealy_store.txt','a')
        # vector_to_file(matrix_to_log_v(self.env_W), self.folder + 'weights.csv','a')
        vector_to_file(matrix_to_log_v(self.env_J), self.folder + 'Jitter_store.txt','a')
        vector_to_file(matrix_to_log_v(self.env_L), self.folder + 'loss_store.txt','a')
        # vector_to_file(matrix_to_log_v(self.env_Rn), self.folder + 'Rn.csv','a')
        vector_to_file(matrix_to_log_v(self.env_Bw), self.folder + 'bandwidth.txt','a')
        vector_to_file(matrix_to_omnet_v(self.env_T), 'ts.txt', 'w')

        new_state = rl_state(self) 

        self.log_everything()
        # return new status and reward
        return new_state, reward, 0


    def easystep(self, action):
        self.counter += 1

        self.upd_env_R_from_R(action)

        # write to file input for Omnet: Routing
        vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
        # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # execute omnet
        omnet_wrapper(self)

        # read Omnet's output: Delay and Lost
        om_output = file_to_csv(self.folder + OMDELAY)
        self.upd_env_D(csv_to_matrix(om_output, self.ACTIVE_NODES))
        self.upd_env_L(csv_to_lost(om_output))

        reward = rl_reward(self)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARDLOG, 'a')
        cur_state = rl_state(self)
        log = np.concatenate(([self.counter], [self.env_L], [np.mean(matrix_to_rl(self.env_D))], [np.max(matrix_to_rl(self.env_D))], [(np.mean(matrix_to_rl(self.env_D)) + np.max(matrix_to_rl(self.env_D)))/2], [stats.gmean(matrix_to_rl(self.env_D))]))
        vector_to_file(log, self.folder + WHOLELOG, 'a')

        # generate traffic for next iteration
        self.upd_env_T(self.tgen.generate())
        # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE', 'DIR'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        new_state = rl_state(self)
        # return new status and reward
        return new_state, reward, 0


    def end(self):
        return
