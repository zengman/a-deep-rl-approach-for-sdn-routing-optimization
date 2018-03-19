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
step = 0
# avg-delay,jitter,loss
def avg(filename,folder):
    df = pd.read_csv(folder+filename, header=None, sep=',')
    d = np.asarray(df)
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

def all_print(folder):
    # filename = 'all_avg.txt'
    avg(DELAY,folder)
    avg(JITTER,folder)
    avg(LOSS,folder)
    avg(REWARD,folder)
    # vector_to_file(avg(DELAY), folder + filename, 'a')
    # vector_to_file(avg(JITTER), folder + filename, 'a')
    # vector_to_file(avg(LOSS), folder + filename, 'a')
    step = avg_band(folder)
    print(step)
all_print(folder)


    






 

