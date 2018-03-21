import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import sys

x_list = [5,10,15,20,25,30,35,40,45,50]



# atype = sys.argv[1]
filename = 'rewardLog.txt'
def get_reward(atype,folder):
    rewardlist = []
    for i in range(10):
        foldername = folder + str(i) + '_' + str(atype) + '/'
        df = pd.read_csv(foldername+filename, header=None, sep=',')
        rewardlist.append(df.head(1))
    return np.asarray(rewardlist)
folderg = './topo/germany50/fd'
# ygt_list = get_reward('t',folderg)
ygu_list = get_reward('u',folderg)
foldert = './topo/ta2/fd'
# ytt_list = get_reward('t',foldert)
# ytu_list = get_reward('u',foldert)
folderi = './new_action/fd'
yiu_list = get_reward('u',folderi)


# yt_list = [4.74265766, 4.73842144, 4.70539904, 4.7169137, 4.7180481, 4.71508884, 4.71711493, 4.72274113, 4.72540665, 4.71796036]
# yu_list = [4.74265766, 4.73571253, 4.71229267, 4.71198416, 4.71785021, 4.71164083, 4.71827602, 4.71713877, 4.72053671, 4.72319841]

ydrl_list = []
ydrl_list.append(4.80013351095) # 5
ydrl_list.append(4.780445357475)  # 10
ydrl_list.append(4.768496345) # 15
ydrl_list.append(4.814918592293144)# 20
ydrl_list.append(4.825982964125) # 25
ydrl_list.append(4.918607345949999) # 30
ydrl_list.append(4.8480342651) # 35
ydrl_list.append(4.761277426625) # 40
ydrl_list.append(4.773210077676768) # 45
ydrl_list.append(4.7922669860975615) # 50
ygdiff_list = []
for i in range(10):
    ygdiff_list.append((ydrl_list[i]-ygu_list[i])/ygu_list[i])
# ygdiff_list = (ydrl_list - ygu_list )/ygu_list
# ydrl_list.append()
# ydrl_list = [,4.783476308490566,4.7684861658571425,4.72175884,0,0,0,0,0,0]
# print(yt_list)
# print(yu_list)
# draw
plt.figure(figsize=(10, 6))
line1 = plt.plot(x_list, ygdiff_list, linestyle=':' , marker='v',color="black", linewidth=2.3, label=' DRL-utili ')
# line2 = plt.plot(x_list, yt_list, linestyle='--', marker='d',color="blue", linewidth=2.3, label=' Throughput ')
# line3 = plt.plot(x_list, yu_list, linestyle='-', marker='o',color="green", linewidth=2.3, label=' Utility ')

plt.xlabel("Number of Flows",fontsize=15)
plt.ylabel("Mean MOS Value",fontsize=15)

plt.rc('legend', fontsize=13)    # legend fontsize
# plt.title(u"xx")
# plt.rc('figure', titlesize=BIGGER_SIZE) 

plt.xlim(5,50) # 限定横轴的范围
plt.ylim(4.70, 4.95) # 限定纵轴的范围
plt.legend()
foo_fig = plt.gcf() # 'get current figure'
foo_fig.savefig('pic1.eps', format='eps', dpi=1000)
plt.show()

