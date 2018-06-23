import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import sys


'''
根据rewardLog信息画图，查看reward值得趋势
'''
# folder = 'runs//t1529463641183343/' # 标记这个文件！！！！
# ----------------------------
folder = 'runs//t1529564305948923/'
# folder = '' 
# atype = sys.argv[1]
filename = 'rewardLog.txt'
df = None
def get_reward():
    rewardlist = []
    global df
    df = pd.read_csv(folder +filename, header=None, sep=',')
    return np.asarray(df)
    # return np.asarray(rewardlist)
    # return rewardlist
ylist = get_reward()
x_list = range(df.iloc[:,0].size)
    

# yu_list = get_reward('u')
# yt_list = [4.74265766, 4.73842144, 4.70539904, 4.7169137, 4.7180481, 4.71508884, 4.71711493, 4.72274113, 4.72540665, 4.71796036]
# yu_list = [4.74265766, 4.73571253, 4.71229267, 4.71198416, 4.71785021, 4.71164083, 4.71827602, 4.71713877, 4.72053671, 4.72319841]

plt.figure(figsize=(20, 6))
ylist = [4.5614163009299995,
4.56935153442,
4.56309486063,
4.56487985897,
4.60607114622,
4.58720303339]
x_list = range(6)
line1 = plt.plot(x_list, ylist, linestyle='-' ,color="black", linewidth=1, label=' DRL ')

plt.xlabel("Number of Flows",fontsize=15)
plt.ylabel("Mean MOS Value",fontsize=15)

plt.rc('legend', fontsize=13)    # legend fontsize
# plt.title(u"xx")
# plt.rc('figure', titlesize=BIGGER_SIZE) 

plt.xlim(0,len(x_list)) # 限定横轴的范围
plt.ylim(min(ylist), min(ylist)+0.05) # 限定纵轴的范围
print(np.mean(ylist))
plt.legend()
foo_fig = plt.gcf() # 'get current figure'
# foo_fig.savefig('pic2.png', format='png', dpi=1000)
plt.show()

