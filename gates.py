import pandas as pd
import numpy as np

WEIGHT = './topo/germany50/link_bandwidth.csv'
def read_weight():
    df = pd.read_csv(WEIGHT, header=None, sep=',')
    return np.asarray(df)
weight = read_weight()
node = 65
use_weight = np.full((node,node),-1,dtype=int)
filename = './topo/germany50/ta2.txt'
def replace_daterate(filename, oldstr,newstr):
    file_data = ""
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            if oldstr in line:
                line = line.replace(oldstr,newstr)
            file_data += line
    with open(filename,'w',encoding='utf-8') as f:
        f.write(file_data)
    f.close()
for i in range(node):
    for j in range(node):
        if weight[i][j] != 0 and use_weight[i][j] == -1:
            print('i = '+str(i)+' j='+str(j)+' v = '+str(weight[i][j]))
            use_weight[i][j] = weight[i][j]
            use_weight[j][i] = weight[i][j]
            oldstr = "sdnnode["+str(i)+"].port++ <--> Channel <--> sdnnode["+str(j)+"].port++;"
            newstr = "sdnnode["+str(i)+"].port++ <--> Channel{datarate="+str(weight[i][j])+"Mbps;} <--> sdnnode["+str(j)+"].port++;"
            # replace_daterate(filename, oldstr,newstr)

  