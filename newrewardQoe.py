
import numpy as np
import math
def reward_QoE(num,bw,delay,jitter,plr,flows):
    print('band')
    print(bw)
    print('delay')
    print(delay)
    print('jitter')
    print(jitter)
    print('plr')
    print(plr)
    a= delay.min()
    b= delay.max()
    j_max = jitter.max()
    mos_value= np.zeros(num)
    for i in range(num):
        penalty  = []
        # print(plr)
        for j in range(len(plr)):
            penalty.append(math.exp(plr[j]) /3)
        penalty = np.asarray(penalty)
        # penalty = math.exp(plr)/3 #%最大为0.9
        va=(delay[i]-a)/(b-a)*0.3 #%最大为0.3
        jit = jitter[i]/j_max*0.2# %最大为0.2
        mos_value[i]= 5.2 * math.log(bw[i]+1) / math.log(flows[i][4]+1)# %flows(i,3)w为流的上界
        mos_temp = mos_value[i]- va - penalty - jit
        # print('mos_temp',mos_temp)
        mos_temp_1= max(mos_temp.max(),0) #%mos值不会小于0
        mos_value[i]= min(5,mos_temp_1) #%mos值最大为5
        # # bw_af_plr = bw[i] * (1 - plr[i]) 
        # bw_af_plr = bw[i] 
        # va = (delay[i]-a) / (b-a) / 15 
        # # noise = jitter[i]./3000*rand(1) 
        # if bw_af_plr > 60:
        #     mos_value[i] = 5 
        # elif bw_af_plr > 50:
        #     mos_value[i] = 4.9+0.1*(bw_af_plr-50)/10 
        # elif bw_af_plr >40:
        #     mos_value[i]=4.75+0.15*(bw_af_plr-40)/10 
        # elif bw_af_plr >30:
        #     mos_value[i]=4.55+0.2*(bw_af_plr-30)/10 
        # elif bw_af_plr >20:
        #     mos_value[i]=4.3+0.25*(bw_af_plr-20)/10 
        # elif bw_af_plr >10:
        #     mos_value[i]=4+0.4*(bw_af_plr-10)/10 
        # elif bw_af_plr >5:
        #     mos_value[i]=3.6+0.3*(bw_af_plr-5)/5 
        # elif bw_af_plr >3:
        #     mos_value[i]=3.3+0.3*(bw_af_plr-3)/5 
        # else:
        #     mos_value[i]=1 
        
        # if  mos_value[i]>1:
        #     mos_value[i]= mos_value[i] - va  
    print('reward')    
    print(mos_value)
    reward = np.mean(mos_value)
    print('均值： ', reward)
    
    return reward