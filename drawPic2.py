import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import sys
import os

x_list = [5,10,15,20,25,30,35,40,45,50]


def file_name(file_dir): 

    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        return dirs #当前路径下所有子目录  
        # return files #当前路径下所有非目录子文件
# atype = sys.argv[1]
def filelsit(file_dir):
    for root, dirs, files in os.walk(file_dir):  
        # print(root) #当前目录路径  
        # return dirs #当前路径下所有子目录  
        return files #当前路径下所有非目录子文件
# atype = sys.argv[1]

filename = 'rewardLog.txt'
mainfolder = sys.argv[1] + '/'
namelist = file_name(mainfolder)
def get_reward(atype,namelist, type):
    rewardlist = []
    for i in range(len(namelist)):
        # foldername = folder + str(i) + '_' + str(atype) + '/'
        foldername = mainfolder + namelist[i] + '/'
        if type in foldername:
            df = pd.read_csv(foldername+filename, header=None, sep=',')
            rewardlist.append(df.head(1))
    return np.asarray(rewardlist)

ygu_list = get_reward('u',namelist, 'uuuu')
ytu_list  = get_reward('t', namelist, 'ttt')
drl_list = []
drlfolder = sys.argv[2] + '/'
flist = filelsit(drlfolder)
for i in range(10):
    for f in flist:
        if f == str((i+1)*5) + '.txt':
            df = pd.read_csv( drlfolder + f, header = None, sep=',')
            # print(f,np.max(df))
            drl_list.append(np.max(df))

plt.figure(figsize=(10, 7))
line1 = plt.plot(x_list, ygu_list, linestyle=':' , marker='v',color="black", linewidth=2.3, label='uu')
line2 = plt.plot(x_list, ytu_list, linestyle='--', marker='d',color="blue", linewidth=2.3, label='tt ')
line3 = plt.plot(x_list, drl_list, linestyle='-', marker='o',color="green", linewidth=2.3, label='drl')

plt.xlabel("Number of Flows",fontsize=15)
plt.ylabel("MOS Improvement ",fontsize=15)

plt.rc('legend', fontsize=13)    # legend fontsize
# plt.title(u"xx")
# plt.rc('figure', titlesize=BIGGER_SIZE) 

plt.xlim(5,50) # 限定横轴的范围
print(np.min(drl_list))
ymin = min(ygu_list) if min(ytu_list) > min(ygu_list) else min(ytu_list)
ymax = max(ygu_list) if max(ytu_list) < max(ygu_list) else max(ytu_list)
ymin = np.min(drl_list) if np.min(drl_list)< ymin else ymin
ymax = np.max(drl_list) if np.max(drl_list) > ymax else ymax
plt.ylim(ymin, ymax) # 限定纵轴的范围
plt.legend()
foo_fig = plt.gcf() # 'get current figure'
# foo_fig.savefig('pic2.eps', format='eps', dpi=1000)
plt.show()

