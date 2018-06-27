import pandas as pd
import numpy as np
mainfolder = './topo/india35/'
# WEIGHT = mainfolder+'link_bandwidth.csv'
# def read_weight():
#     df = pd.read_csv(WEIGHT, header=None, sep=',')
#     return np.asarray(df)
# weight = read_weight()
node = 35
flow_num = 100

'''
输入folder， 节点个数，流的个数，生成流量矩阵
'''
def genFlowsCSV(folder,node,flow_num,csvname):
    
    flow_tong = int(flow_num * 0.25) + 1
    src_dest = np.random.randint(1,node,size=[flow_num,2])
    # 2 step 生成df 10-50
    recode = []
    one = np.random.randint(1,node,size=[5,2]) # 新生成5条
    for i in range(5):
        while one[i][0] == one[i][1]:
            a = np.random.randint(1,node,size=[1,2])
            one[i][0] = a[0][0]
            one[i][1] = a[0][1]
        src_dest[i][0] = one[i][0]
        src_dest[i][1] = one[i][1]
        if i == 3 or i == 4:
            temp = []
            temp.append(src_dest[i][0])
            temp.append(src_dest[i][1])
            recode.append(temp)
    cnt = 5
    for j in range(9):
        # 每次新生成3条，并且取前面的最后2条重复
        src_dest[cnt][0] = src_dest[cnt-1][0]
        src_dest[cnt][1] = src_dest[cnt-1][1]
        cnt += 1
        # src_dest[cnt][0] = src_dest[cnt-3][0]
        # src_dest[cnt][1] = src_dest[cnt-3][1]
        # cnt += 1
        # src_dest[cnt][0] = src_dest[cnt-5][0]
        # src_dest[cnt][1] = src_dest[cnt-5][1]
        # cnt += 1
        buchognfuliu = 4
        two = np.random.randint(1,node,size=[buchognfuliu,2])
        recode = []
        for i in range(buchognfuliu):
            while two[i][0] == two[i][1]:
                a = np.random.randint(1,node,size=[1,2])
                two[i][0] = a[0][0]
                two[i][1] = a[0][1]
            src_dest[cnt][0] = two[i][0]
            src_dest[cnt][1] = two[i][1]
            cnt += 1
        

    df = np.random.randint(0,3,size=[flow_num,1])
    # 3 step 生成Df, 中间差值不要超过5
    Df = np.random.randint(10,15, size=[flow_num,1]) + df
    #字典中的key值即为csv中列名
    columns = ['src','dest','df','DF']
    dataframe = pd.DataFrame({'DF':Df[:,0],'dest':src_dest[:,1],'df':df[:,0],'src':src_dest[:,0] })
    #将DataFrame存储为csv,index表示是否显示行名，default=True
    # filename = ''
    # dataframe.to_csv(filename,index=False,sep=',',header=False,columns=columns)
    dataframe.to_csv(csvname,index=False,sep=',',header=False,columns=columns)
    
    genFlowsCSV(mainfolder, node, flow_num,'test.csv')

    

  