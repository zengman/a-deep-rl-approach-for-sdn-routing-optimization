import pandas as pd
import numpy as np

WEIGHT = './new_action/link_bandwidth.csv'
def read_weight():
    df = pd.read_csv(WEIGHT, header=None, sep=',')
    return np.asarray(df)
weight = read_weight()
for i in range(35):
    for j in range(35):
        if weight[i][j] != 0:
            print('i = '+str(i)+' j='+str(j)+' v = '+str(weight[i][j]))

    