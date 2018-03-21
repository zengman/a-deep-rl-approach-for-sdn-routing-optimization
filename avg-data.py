import numpy as np
import pandas as pd
import sys
# from Environment import vector_to_file
# from helper import pretty
DELAY = 'Dealy_store.txt'
BANDWIDTH ='bandwidth_store.txt'
JITTER='Jitter_store.txt'
LOSS = 'loss_store.txt'
REWARD ='rewardLog.txt'
number = sys.argv[1]
folder = 'runs/'+str(number)+'/'
# folder = 'new_action/'+str(number)+'/'
flow_num = 5
step = sys.argv[2]
# avg-delay,jitter,loss
def avg(filename,folder,step):
    df = pd.read_csv(folder+filename, header=None, sep=',')
    if int(step) == 0:
        d = np.asarray(df)
    else:
        step = int(step)
        d = np.asarray(df.head(step))
    # string = str(d.mean())
    print(d.mean())
    
    # return d.mean()
    
# band sum for every flow
def avg_band(folder):
    df = pd.read_csv(folder+BANDWIDTH, header=None, sep=',')
    d = np.asarray(df)
    length = len(df)
    # step = length
    print(d.sum()/length)
    return length

def all_print(folder,step):
    # filename = 'all_avg.txt'
    avg(DELAY,folder,step)
    avg(JITTER,folder,step)
    avg(LOSS,folder,step)
    avg(REWARD,folder,step)
    # vector_to_file(avg(DELAY), folder + filename, 'a')
    # vector_to_file(avg(JITTER), folder + filename, 'a')
    # vector_to_file(avg(LOSS), folder + filename, 'a')
    stepp = avg_band(folder)
    print(step)
    print(stepp)
all_print(folder,step)


    






 

