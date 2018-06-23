# -*- coding:utf-8 -*-

"""
绘制箱体图
Created on 2017.09.04 by ForestNeo
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
generate data from min to max
"""
import os  
      
def file_name(file_dir): 

    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        return dirs #当前路径下所有子目录  
        # print(files) #当前路径下所有非目录子文件
        # break
# df2 = pd.read_csv ("testfoo.csv" , encoding = "utf-8")
# print (df2)

def get_reward(folder):
    rewardlist = []
    filename = 'rewardLog.txt'
    df = pd.read_csv(folder +filename, header=None, sep=',')
    a = np.asarray(df) 
    for x in a:
        # summ += x[0]
        # cnt += 1
        # if cnt > 10:
        #     rewardlist.append(summ/cnt)
        #     summ = 0
        #     cnt = 1
        rewardlist.append(x[0])
    # print(rewardlist)
    return rewardlist

def list_generator(number, min, max):
    dataList = list()
    for i in range(1, number):
        dataList.append(np.random.randint(min, max)) # 生成number 个 值，值范围在min,max
    return dataList

#generate 4 lists to draw
list1 = list_generator(100, 20, 80)
list2 = list_generator(100, 20, 50)
list3 = list_generator(100, 50, 100)
list4 = list_generator(100, 5, 60)
# print('---------')
# # print(list1)
# ff = ['runs//t1529551001737516/', 'runs//t1529551164275400/', 'runs//t1529551424749523/']
Dataflst = {
    # "dataSet1":list1,
    # "dataSet2":list2,
    # "dataSet3":list3,
    # "dataSet4":list4,
    # "dataSet5":list5,
    # "dataSet6":list6,
    # "dataSet7":list7,
    # "dataSet8":list8,
    # "dataSet9":list9
}
# string = "t1529561445445078  t1529561883107786  t1529562512211731  t1529563318028046  t1529564305948923  t1529565454766303  t1529566739612419  t1529569067521769"

# stry = string.split('  ')
datastore = {}
for i in range(10):
    datastore[i] = []
'''
第一次和第二次的reward保存
{0: [4.5614163009299995, 4.51988306536], 1: [4.56935153442, 4.6118146221], 2: [4.56309486063, 4.56484574076], 3: [4.56487985897, 4.580075425179999], 4: [4.60607114622, 4.60958692], 5: [4.58720303339, 4.554090894740001], 6: [4.593363289700001, 4.514249339100001], 7: [4.58647096322, 4.5907398937999995], 8: [4.57492670448, 4.552969622999999], 9: [4.523090301480001, 4.66546572518]}
'''

strylist = file_name("runs_temp")
leng = int(len(strylist)/10)
for i in range(leng):
    for j in range(10):
        reward = get_reward("runs_temp//"+strylist[i*10+j]+"/")
        avg = np.mean(reward)
        datastore[j].append(avg)

server_flow = file_name("runs")
leng = int(len(server_flow)/10)
for i in range(len(server_flow)):
    ffile = server_flow[i]
    reward = get_reward("runs//"+ffile+"/")
    avg = np.mean(reward)
    
    start = ffile.find('m_') + 2
    end = len(ffile)
    
    flownume = int(ffile[start:end])/5-1
    # print(flownume)
    datastore[flownume].append(avg)
    

server_flow = file_name("runs_2")
leng = int(len(server_flow)/10)
for i in range(len(server_flow)):
    ffile = server_flow[i]
    reward = get_reward("runs_2//"+ffile+"/")
    avg = np.mean(reward)
    start = ffile.find('m_') + 2
    end = len(ffile)
    
    flownume = int(ffile[start:end])/5-1
    # print(flownume)
    datastore[flownume].append(avg)
# print(datastore)
# list1 = get_reward(ff[0])
# list2 = get_reward(ff[1])
# list3 = get_reward(ff[2])
# list1 = get_reward('runs//t1529563252295149/')
# f=['runs//t1529552104546138/', 'runs//t1529552521940175/', 'runs//t1529552986530861/', 'runs//t1529553501039957/', 'runs//t1529554055372014/', 'runs//t1529554661387432/', 'runs//t1529555339061735/']
# list2 = get_reward('runs//t1529500672772215/')
# list4 = get_reward(f[0])
# list5 = get_reward(f[1])
# list6 = get_reward(f[2])
# list7 = get_reward(f[3])
# list8 = get_reward(f[4])
# list9 = get_reward(f[5])
# list1 = [4.5534336825, 4.544376590000001, 4.551941349000001, 4.543318294, 4.569294521499999, 4.6665929995, 4.507183908, 4.536584164500001, 4.586533831000001, 4.59995823, 4.5391521975, 4.540309143, 4.565350726000001, 4.526679034499999, 4.584684511500001, 4.578305034000001, 4.5741333945, 4.550934004999999, 4.522883510500001, 4.582474206500001]

# Dataflst['dataSet2'] = list1
data = pd.DataFrame(datastore)

#draw
data.boxplot()
plt.ylabel("ylabel")
plt.xlabel("different datasets")
plt.show()

