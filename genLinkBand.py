import pandas as pd
import numpy as np
mainfolder = './topo/india35/'
# WEIGHT = mainfolder+'link_bandwidth.csv'
# def read_weight():
#     df = pd.read_csv(WEIGHT, header=None, sep=',')
#     return np.asarray(df)
# weight = read_weight()
node = 35
flow_num = 50

'''
输入folder， 节点个数，流的个数，生成流量矩阵
'''
def genFlowsCSV(folder,node,flow_num):
    filename = folder + 'flow_test.csv'
    # a=np.random.randint(10,100,size=[node,node]) 
    # # 1 step 生成流的源和目的
    src_dest = np.random.randint(1,node,size=[flow_num,2])
    # 2 step 生成df 10-50
    for i in range(flow_num):
        while src_dest[i][0] == src_dest[i][1]:
            a = np.random.randint(1,node,size=[1,2])
            src_dest[i][0] = a[0][0]
            src_dest[i][1] = a[0][1]
    df = np.random.randint(5,20,size=[flow_num,1])
    # 3 step 生成Df, 中间差值不要超过5
    Df = np.random.randint(0,10, size=[flow_num,1]) + df
    #字典中的key值即为csv中列名
    columns = ['src','dest','df','DF']
    dataframe = pd.DataFrame({'DF':Df[:,0],'dest':src_dest[:,1],'df':df[:,0],'src':src_dest[:,0] })
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv(filename,index=False,sep=',',header=False,columns=columns)

# genFlowsCSV(mainfolder, node, flow_num)
WEIGHT = './topo/india35/link_bandwidth.csv'
def read_weight():
    df = pd.read_csv(WEIGHT, header=None, sep=',')
    return np.asarray(df)
weight = read_weight()
weightflow = np.full([node]*2, 0.0, dtype=float) 

def getlinkuiti(pathset,bw,flow_num,node):
    # print(pathset)
    for j in range(flow_num):
        path = pathset[j]
        for i in range(len(path)-1):
            cur = path[i]
            next = path[i+1]
            weightflow[cur][next] += bw[j]
            weightflow[next][cur] += bw[j]
    for i in range(node):
        for j in range(node):
            if weightflow[i][j]!= 0 and weight[i][j]!=0:
                weightflow[i][j] = weightflow[i][j]/weight[i][j]
    
    #columns = ['src','dest','df','DF']
    dataframe = pd.DataFrame(weightflow)
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    dataframe.to_csv('testbandlink.csv',index=False,sep=',',header=False)


# node = 35
use_weight = np.full((node,node),-1,dtype=int)
filename = './topo/india35/india35.txt'
newfile = './topo/india35/india35new.txt'
def replace_daterate(filename, oldstr,newstr):
    file_data = ""
    with open(filename,'r',encoding='utf-8') as f:
        for line in f:
            
            if line[51] == "k":
                number = int(line[50:51])
                oldstr = "Channel_c{datarate="+line[50:51]
                newstr = "Channel_c{datarate="+str(number)

            else:
                number = int(line[50:52])
                oldstr = "Channel_c{datarate="+line[50:52]
                newstr = "Channel_c{datarate="+str(number)
            # print(oldstr + ", "+newstr)
            # if oldstr in line:
            line = line.replace(oldstr,newstr)
            # print(line.replace(oldstr,newstr))
            file_data += line
    with open(newfile,'w',encoding='utf-8') as f:
        f.write(file_data)
    f.close()

def main():
    for i in range(node):
        for j in range(node):
            if weight[i][j] != 0 and use_weight[i][j] == -1:
                #print('i = '+str(i)+' j='+str(j)+' v = '+str(weight[i][j]))
                use_weight[i][j] = weight[i][j]
                use_weight[j][i] = weight[i][j]
                # oldstr = "sdnnode["+str(i)+"].port++ <--> Channel{datarate="+str(weight[i][j]-5)+"kbps;} <--> sdnnode["+str(j)+"].port++;"
                # newstr = "sdnnode["+str(i)+"].port++ <--> Channel{datarate="+str(weight[i][j]-5)+"kbps;} <--> sdnnode["+str(j)+"].port++;"
                # replace_daterate(filename, oldstr,newstr)
    
  

    

  