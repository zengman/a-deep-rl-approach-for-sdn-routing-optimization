"""
Environment.py
"""
__author__ = "giorgio@ac.upc.edu"

import numpy as np
from scipy import stats
import subprocess
import networkx as nx
import traceback

from helper import pretty, softmax
from Traffic import Traffic
from Reward_QoE import reward_QoE 


OMTRAFFIC = 'Traffic.txt'
OMBALANCING = 'Balancing.txt'
OMROUTING = 'Routing.txt'
OMDELAY = 'Delay.txt'
OMDBW = 'Bandwidth.txt'
OMPLR = 'PacketLossRate.txt'
OMJITTER = 'Jitter.txt'


TRAFFICLOG = 'TrafficLog.csv'
BALANCINGLOG = 'BalancingLog.csv'
REWARDLOG = 'rewardLog.csv'
WHOLELOG = 'Log.csv'
OMLOG = 'omnetLog.csv'


# FROM MATRIX
def matrix_to_rl(matrix):
    return matrix[(matrix!=-1)]

matrix_to_log_v = matrix_to_rl

def matrix_to_omnet_v(matrix):
    return matrix.flatten()

def vector_to_file(vector, file_name, action):
    string = ','.join(pretty(_) for _ in vector)
    with open(file_name, action) as file:
        return file.write(string + '\n')


# FROM FILE
def file_to_csv(file_name):
    # reads file, outputs csv
    with open(file_name, 'r') as file:
        return file.readline().strip().strip(',')

def csv_to_matrix(string, nodes_num):
    # reads text, outputs matrix
    v = np.asarray(tuple(float(x) for x in string.split(',')[:nodes_num**2]))
    M = np.split(v, nodes_num)
    return np.vstack(M)

def csv_to_metrics(string):
    return float(string.split(',')[-1])


# FROM RL
def rl_to_matrix(vector, nodes_num):
    M = np.split(vector, nodes_num)
    for _ in range(nodes_num):
        M[_] = np.insert(M[_], _, -1)
    return np.vstack(M)


# TO RL
def rl_state(env):
    if env.STATUM == 'RT':
        return np.concatenate((matrix_to_rl(env.env_B), matrix_to_rl(env.env_T)))
    elif env.STATUM == 'T':
        return matrix_to_rl(env.env_T)
    elif env.STATUM == 'F':
        return np.concatenate((matrix_to_rl(env.env_B), matrix_to_rl(env.env_D), matrix_to_rl(env.env_J), matrix_to_rl(env.env_L)))

def rl_reward(env, model, data_mean, data_std):

    delay = env.env_D
    np.asarray(env.env_D)
    mask = delay == np.inf
    delay[mask] = len(delay)*np.max(delay[~mask])

    if env.PRAEMIUM == 'AVG':
        # reward = -np.mean(matrix_to_rl(delay))
       reward = reward_QoE (env.env_B,env.env_D,env.env_J,env.env_L, model, data_mean, data_std)
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
        omnet_output = subprocess.check_output([simexe, '-n', simfolder, simini, env.folder + 'folder.ini']).decode()
    except Exception as e:
        print(e.args)
        #  omnet_output = e.stdout.decode()

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



# label environment
class OmnetLinkweightEnv():

    def __init__(self, DDPG_config, folder):
        self.ENV = 'label'
        self.ROUTING = 'Linkweight'

        self.folder = folder

        self.ACTIVE_NODES = DDPG_config['ACTIVE_NODES']
        self.flow_num = DDPG_config['FLOW_NUM']

        self.ACTUM = DDPG_config['ACTUM']

        topology = 'omnet/router/NetworkAll.matrix'
        self.graph = nx.Graph(np.loadtxt(topology, dtype=int))
        if self.ACTIVE_NODES != self.graph.number_of_nodes():
            return False
        ports = 'omnet/router/NetworkAll.ports'
        self.ports = np.loadtxt(ports, dtype=int)
        # 链路的weight，也是输出维度 action
        self.a_dim = self.graph.number_of_edges()
        # +flow_num
        # 输入维度，状态输入，traffic矩阵 state
        self.s_dim = self.flow_num*4
        # self.ACTIVE_NODES**2 - self.ACTIVE_NODES    # traffic minus diagonal
        # self.flow_num

        self.STATUM = DDPG_config['STATUM']
        if self.STATUM == 'RT':
            self.s_dim *= 2    # traffic + routing table minus diagonals

        self.PRAEMIUM = DDPG_config['PRAEMIUM']

        capacity = self.ACTIVE_NODES * (self.ACTIVE_NODES -1)

        self.TRAFFIC = DDPG_config['TRAFFIC']
        self.tgen = Traffic(self.ACTIVE_NODES, self.TRAFFIC, capacity, self.flow_num)

        self.CLUSTER = DDPG_config['CLUSTER'] if 'CLUSTER' in DDPG_config.keys() else False

        self.env_T = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # traffic
        self.env_W = np.full([self.a_dim], -1.0, dtype=float)           # weights
        self.env_R = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)    # routing
        self.env_Rn = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=int)   # routing (nodes)
        # self.env_D = np.full([self.ACTIVE_NODES]*2, -1.0, dtype=float)  # delay
        self.env_B = np.full([self.flow_num], -1.0, dtype=float)#Bandwidth
        self.env_D = np.full([self.flow_num], -1.0, dtype=float)  # delay
        self.env_J = np.full([self.flow_num], -1.0, dtype=float)  # Jitter
        self.env_L = np.full([self.flow_num], -1.0, dtype=float)  # lost packets
        # self.env_L = -1.0  # lost packets
        self.counter = 0


    def upd_env_T(self, matrix):
        self.env_T = np.asarray(matrix)
        # np.fill_diagonal(self.env_T, -1)

    def upd_env_W(self, vector):
        self.env_W = np.asarray(softmax(vector))

    #更新路由 dijstra
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

        
    def upd_env_B(self, matrix):
        # 列表转换为矩阵
        self.env_D = matrix
        # np.asarray(matrix)
        # np.fill_diagonal(self.env_D, -1)

    def upd_env_D(self, matrix):
        # 列表转换为矩阵
        self.env_D = matrix
        # np.asarray(matrix)
        # np.fill_diagonal(self.env_D, -1)


    def upd_env_J(self, matrix):
        # 列表转换为矩阵
        self.env_J = matrix
        #  np.asarray(matrix)
        # np.fill_diagonal(self.env_J, -1)

    def upd_env_L(self, matrix):
        self.env_L = matrix


    def logheader(self, easy=False):
        nice_matrix = np.chararray([self.ACTIVE_NODES]*2, itemsize=20)
        for i in range(self.ACTIVE_NODES):
            for j in range(self.ACTIVE_NODES):
                nice_matrix[i][j] = str(i) + '-' + str(j)
        np.fill_diagonal(nice_matrix, '_')
        nice_list = list(nice_matrix[(nice_matrix!=b'_')])
        th = ['t' + _.decode('ascii') for _ in nice_list]
        rh = ['r' + _.decode('ascii') for _ in nice_list]
        dh = ['d' + _.decode('ascii') for _ in nice_list]
        ah = ['a' + str(_[0]) + '-' + str(_[1]) for _ in self.graph.edges()]
        header = ['counter'] + th + rh + dh + ['lost'] + ah + ['reward']
        if easy:
            header = ['counter', 'lost', 'AVG', 'MAX', 'AXM', 'GEO']
        vector_to_file(header, self.folder + WHOLELOG, 'w')


    def render(self):
        return


    def reset(self, easy=False):
        if self.counter != 0:
            return None

        self.logheader(easy)

        # routing
        self.upd_env_W(np.full([self.a_dim], 0.50, dtype=float))
        self.upd_env_R()
        if self.ACTUM == 'DELTA':
            vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
            # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # traffic
        self.upd_env_T(self.tgen.generate())

        vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        return rl_state(self)


    def step(self, action, model, data_mean, data_std):
        self.counter += 1

        self.upd_env_W(action)
        self.upd_env_R()

        # write to file input for Omnet: Routing
        vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
        # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # execute omnet
        omnet_wrapper(self)

        # read Omnet's output: Delay and Lost
        om_output_bandwidth = file_to_csv(self.folder + OMDBW)
        om_output_delay = file_to_csv(self.folder + OMDELAY)
        om_output_jitter = file_to_csv(self.folder + OMJITTER)
        om_output_plr = file_to_csv(self.folder + OMPLR)
        self.upd_env_B(csv_to_metrics(om_output_bandwidth))
            # csv_to_matrix(om_output, self.flow_num)) 
        self.upd_env_D(csv_to_metrics(om_output_delay))
        # (csv_to_matrix(om_output, self.flow_num))
        self.upd_env_J(csv_to_metrics(om_output_jitter))
        # (csv_to_matrix(om_output, self.flow_num))
        #  self.ACTIVE_NODES))
        self.upd_env_L(csv_to_metrics(om_output_plr))
        # csv_to_lost(om_output))

        reward = rl_reward(self, model, data_mean, data_std)

        # log everything to file
        vector_to_file([-reward], self.folder + REWARDLOG, 'a')
        cur_state = rl_state(self)
        log = np.concatenate(([self.counter], matrix_to_log_v(self.env_T), matrix_to_log_v(self.env_Rn), matrix_to_log_v(self.env_D), [self.env_L], matrix_to_log_v(self.env_W), [-reward]))
        vector_to_file(log, self.folder + WHOLELOG, 'a')

        # generate traffic for next iteration
        self.upd_env_T(self.tgen.generate())
        # write to file input for Omnet: Traffic, or do nothing if static
        if self.TRAFFIC.split(':')[0] not in ('STAT', 'STATEQ', 'FILE'):
            vector_to_file(matrix_to_omnet_v(self.env_T), self.folder + OMTRAFFIC, 'w')

        new_state = rl_state(self)
        # return new status and reward
        return new_state, reward, 0


    def easystep(self, action, model, data_mean, data_std):
        self.counter += 1

        self.upd_env_R_from_R(action)

        # write to file input for Omnet: Routing
        vector_to_file(matrix_to_omnet_v(self.env_R), self.folder + OMROUTING, 'w')
        # VERIFY FILE POSITION AND FORMAT (separator, matrix/vector) np.savetxt("tmp.txt", routing, fmt="%d")

        # execute omnet
        omnet_wrapper(self)

        # read Omnet's output: Delay and Lost
        om_output_bandwidth = file_to_csv(self.folder + OMDBW)
        om_output_delay = file_to_csv(self.folder + OMDELAY)
        om_output_jitter = file_to_csv(self.folder + OMJITTER)
        om_output_plr = file_to_csv(self.folder + OMPLR)
        self.upd_env_B(csv_to_metrics(om_output_bandwidth))
        # csv_to_matrix(om_output, self.flow_num)) 
        self.upd_env_D(csv_to_metrics(om_output_delay))
        # (csv_to_matrix(om_output, self.flow_num))
        self.upd_env_J(csv_to_metrics(om_output_jitter))
        # (csv_to_matrix(om_output, self.flow_num))
        #  self.ACTIVE_NODES))
        self.upd_env_L(csv_to_metrics(om_output_plr))

        reward = rl_reward(self, model, data_mean, data_std)

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
