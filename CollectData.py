import pandas as pd
import numpy as np
folder = "./runs/t1521265683756629/"
flow_num = 10
cnt = 1000
df_reward = pd.read_csv(folder + 'rewardLog.txt', header=None, sep=' ')
reward = df_reward.mean()
print('reward = ',reward[0])
df_delay = pd.read_csv(folder + 'Dealy_store.txt', header=None, sep=',')
delay = df_delay.sum().mean()
print('dealy = ',delay)
df_bandwidth = pd.read_csv(folder + 'bandwidth.txt', header=None, sep=',')
arrs = 0
for i in range(89):

    col=df_bandwidth.iloc[i:,]  
    # print('col',col)
    arrs+=col.sum()
    # print('flow =',col.sum())

print('band = ',arrs/89)
print(df_delay.iloc[0][0])






