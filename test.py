import pandas as pd
import numpy as np
i = 6
re = pd.read_csv('new_action/f'+str(i)+'_u/rewardLog.txt', header=None, sep=',')
print(np.asarray(re[0][0]))
df = pd.read_csv('new_action/f'+str(i)+'_u/Delay.txt', header=None, sep=',')
d = np.asarray(df.head(1))
d[0][-1]=0
print(d.mean())
# df2 = pd.read_csv ("testfoo.csv" , encoding = "utf-8")
# print (df2)