import pandas as pd
import numpy as np

'''
对比算法： 读取相应的选择路径和带宽分配等数据
india35
ta2 
germany50

'''
main = './topo/india35/'
PATH = main+'path'
BAND_THR = main+'bw_thr'
# BAND_UTILITY = main+'bw_utiliy'
# PATH_CHOICE_U  = main+'path_chosen_utility'
# PATH_CHOICE_T = main+'path_chosen_thr'
# WEIGHT = main+'link_bandwidth'

FLOW = main+'flows'


class OtherAction():
    def __init__(self, choice, flow_num, id, sort):
        self.id = id
        self.topo = 'data_new_multi/'+ sort +'/' # sort = Mat03...
        self.all_path = pd.read_csv( self.topo + 'path_' + sort + '.csv', header=None, sep=',')
        self.choce_path = []
        self.flow_num = flow_num
        self.bw = []
        self.weight = self.topo + 'link_bandwidth_' + sort + '.csv'
        self.flowcsv = self.topo + 'flows_' + sort + '.csv'
        print(self.flowcsv)
        if choice == 0:
            self.band = self.topo + 'bw_thr_' + sort + '.csv'
            self.path_choce_name = self.topo + 'path_chosen_thr_' + sort + '.csv'
        else:
            self.band = self.topo + 'bw_utiliy_' + sort + '.csv'
            self.path_choce_name = self.topo + 'path_chosen_utility_' + sort + '.csv'
            
    def read_path_choice(self):
        choice  = pd.read_csv(self.path_choce_name, header=None, sep=',')# 第id行数据
        choice_path = []
        id = self.id
        for i in range(self.flow_num):
            
            path_id = choice.iloc[id][i]-1 # indix 从0开始
            path = self.read_path(path_id) 
            # print('path id', path_id)
            # print('readchoce ,', path)
            choice_path.append(path)
        return choice_path

    def read_path(self, path_id):
        path = self.all_path.iloc[path_id,:]
        path = np.asarray(path)-1
        
        return path[path!=-1]
    def read_band(self):
        band = pd.read_csv(self.band, header=None, sep=',')
        t = band.iloc[self.id]
        t = np.asarray(t[:self.flow_num])
        print(t)
        return t

    def read_weight(self):
        df = pd.read_csv(self.weight, header=None, sep=',')
        return np.asarray(df)

    def csv_traffic(self):
        t = np.full((self.flow_num, 4), -1.0, dtype=float)
        df=pd.read_csv(self.flowcsv, header=None, sep=',')
        data = df.head(self.flow_num)
        for i in range(self.flow_num):
            t[i,:] = list(data.ix[i,:])
        a = np.arange(0,self.flow_num)
        t[:,0] -= 1
        t[:,1] -= 1
        # t[:,4] = t[:,4] * 0.2
        # print(t[:,0])
        # print(t[:,1])
        return np.c_[a,t]


    
        
        
        



