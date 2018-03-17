import pandas as pd
import numpy as np

df=pd.read_csv('./flows.csv', header=None, sep=',')
# print(df.head(5)[0][0])
t = np.full((5, 4), -1.0, dtype=int)
data = df.head(5)
print(data[0])
b = list(data[0])
print(b)
for i in range(5):
    b = list(data.ix[i,:])
    t[i,:] = b
    
a = np.arange(0,5)
print(np.c_[a,t])