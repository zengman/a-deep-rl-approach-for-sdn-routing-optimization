import matplotlib.pyplot as plt  
import seaborn as sns  
import pandas as pd
import numpy as np
filename = 'rewardLog.txt'
folder = 'runs//t1530066583454230_Mat1_flow_num_5/'
df = pd.read_csv(folder +filename, header=None, sep=',')
print(np.max(df))
print(np.mean(df))