import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import sys

x_list = [5,10,15,20,25,30,35,40,45,50]
# y_list = range(1,20)

folder = 'new_action/fd'
# atype = sys.argv[1]
filename = 'rewardLog.txt'
def get_reward(atype):
    rewardlist = []
    for i in range(10):
        foldername = folder + str(i) + '_' + str(atype) + '/'
        df = pd.read_csv(foldername+filename, header=None, sep=',')
        rewardlist.append(df.head(1))
    return np.asarray(rewardlist)


yt_list = get_reward('t')
yu_list = get_reward('u')
ydrl_list = [4.76007384330578,4.78295708,4.768407162857143,0,0,0,0,0,0,0]
# print(yt_list)
# print(yu_list)
# draw
plt.figure(figsize=(13, 7))
plt.plot(x_list, yt_list, linestyle='--', marker='o',color="blue", linewidth=2, label=' Throughput ')
plt.plot(x_list, yu_list, linestyle='-', marker='o',color="green", linewidth=2, label=' Utility ')
plt.plot(x_list, ydrl_list, linestyle=':' , marker='o',color="black", linewidth=2, label=' DRL ')

plt.xlabel("Number of Flows")
plt.ylabel("Mean MOS Value")
plt.title(u"xx")
plt.legend((u"图例",))
plt.xlim(5,50)
plt.ylim(4.7, 4.87)
plt.legend()
plt.show()
# plt.savefig('test.png')