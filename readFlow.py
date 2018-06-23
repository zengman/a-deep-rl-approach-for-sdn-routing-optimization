import pandas as pd
import numpy as np

WEIGHT = './topo/germany50/link_bandwidth.csv'
def read_weight():
    df = pd.read_csv(WEIGHT, header=None, sep=',')
    return np.asarray(df)
weight = read_weight()
node = 35
use_weight = np.full((node,node),-1,dtype=int)
filename = './topo/germany50/germany50.txt'
newfile = './topo/germany50/germany50new.txt'
def replace_daterate(filename, oldstr,newstr):
    file_data = ""
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            
            if line[51] == "k":
                number = int(line[50:51])
                oldstr = "Channel_c{datarate="+line[50:51]
                dd = np.random.randint(70,100, size=None)
                newstr = "Channel_c{datarate="+str(dd)

            else:
                number = int(line[50:52])
                oldstr = "Channel_c{datarate="+line[50:52]
                dd = np.random.randint(70,100, size=None)
                newstr = "Channel_c{datarate="+str(dd)
            # print(oldstr + ", "+newstr)
            # if oldstr in line:
            line = line.replace(oldstr,newstr)
            # print(line.replace(oldstr,newstr))
            file_data += line
    with open(newfile,'w',encoding='utf-8') as f:
        f.write(file_data)
    f.close()

for i in range(node):
    for j in range(node):
        if weight[i][j] != 0 and use_weight[i][j] == -1:
            #print('i = '+str(i)+' j='+str(j)+' v = '+str(weight[i][j]))
            use_weight[i][j] = weight[i][j]
            use_weight[j][i] = weight[i][j]
            oldstr = "sdnnode["+str(i)+"].port++ <--> Channel{datarate="+str(weight[i][j]-5)+"kbps;} <--> sdnnode["+str(j)+"].port++;"
            newstr = "sdnnode["+str(i)+"].port++ <--> Channel{datarate="+str(weight[i][j]-5)+"kbps;} <--> sdnnode["+str(j)+"].port++;"
            replace_daterate(filename, oldstr,newstr)

  