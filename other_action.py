import pandas as pd
import numpy as np
main = './new_action/'
PATH = main+'path.csv'
BAND_THR = main+'bw_thr.csv'
BAND_UTILITY = main+'bw_utiliy.csv'
PATH_CHOICE_U  = main+'path_chosen_utility.csv'
PATH_CHOICE_T = main+'path_chosen_thr.csv'
WEIGHT = main+'link_bandwidth.csv'

FLOW = main+'flows.csv'


class OtherAction():
    def __init__(self, choice, flow_num, id):
        self.id = id
        self.all_path = pd.read_csv(PATH, header=None, sep=',')
        self.choce_path = []
        self.flow_num = flow_num
        self.bw = []
        if choice == 0:
            self.band = BAND_THR
            self.path_choce_name = PATH_CHOICE_T
        else:
            self.band = BAND_UTILITY
            self.path_choce_name = PATH_CHOICE_U
    def read_path_choice(self):
        choice  = pd.read_csv(self.path_choce_name, header=None, sep=',')# 第id行数据
        choice_path = []
        id = self.id
        for i in range(self.flow_num):
            
            path_id = choice.iloc[id][i]-1
            path = self.read_path(path_id)
            # print(path_id)
            choice_path.append(path)
        return choice_path

    def read_path(self, path_id):
        path = self.all_path.iloc[path_id,:]
        path = np.asarray(path)-1
        
        return path[path!=-1]
    def read_band(self):
        band = pd.read_csv(self.band, header=None, sep=',')
        t = band.iloc[self.id]
        t = np.asarray(t[t!=0])
        return t

    def read_weight(self):
        df = pd.read_csv(WEIGHT, header=None, sep=',')
        return np.asarray(df)

    def csv_traffic(self):
        t = np.full((self.flow_num, 4), -1.0, dtype=float)
        df=pd.read_csv(FLOW, header=None, sep=',')
        data = df.head(self.flow_num)
        for i in range(self.flow_num):
            t[i,:] = list(data.ix[i,:])
        a = np.arange(0,self.flow_num)
        t[:,0] -= 1
        t[:,1] -= 1
        print(t[:,2])
        return np.c_[a,t]


    
        
        
        



