"""
ddpg.py
"""
__author__ = "giorgio@ac.upc.edu"
__credits__ = "https://github.com/yanpanlau"

# import simulator as environment env()
from Environment import OmnetBalancerEnv
from Environment import OmnetLinkweightEnv
from Environment import vector_to_file
import numpy as np
import tensorflow as tf
import sys
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
from helper import setup_exp, setup_run, parser, pretty, scale
from Reward_QoE import NN_training 

def playGame(DDPG_config, model, data_mean, data_std,train_indicator=1):    #1 means Train, 0 means simply Run
    # SETUP STARTS HERE
    if train_indicator > 0:
        folder = setup_run(DDPG_config)
        print('folder=', folder)
    elif train_indicator == 0:
        folder = DDPG_config['EXPERIMENT']

    if DDPG_config['RSEED'] == 0:
        DDPG_config['RSEED'] = None
    np.random.seed(DDPG_config['RSEED'])

    ACTIVE_NODES = DDPG_config['ACTIVE_NODES']

    # Generate an environment
    # 需要在omnet/router中加入相应的内容，比如拓扑图networkAll.matrix等
    if DDPG_config['ENV'] == 'balancing':
        env = OmnetBalancerEnv(DDPG_config, folder)
    elif DDPG_config['ENV'] == 'label':
        env = OmnetLinkweightEnv(DDPG_config, folder)

    action_dim, state_dim = env.a_dim, env.s_dim
    print('state_dim =',state_dim)
    # action_dim = graph_number_of_edges

    MU = DDPG_config['MU']
    THETA = DDPG_config['THETA']
    SIGMA = DDPG_config['SIGMA']

    ou = OU(action_dim, MU, THETA, SIGMA)       #Ornstein-Uhlenbeck Process
    # 数学上的一个随机过程

    BUFFER_SIZE = DDPG_config['BUFFER_SIZE']
    BATCH_SIZE = DDPG_config['BATCH_SIZE']
    GAMMA = DDPG_config['GAMMA']
    EXPLORE = DDPG_config['EXPLORE']
    EPISODE_COUNT = DDPG_config['EPISODE_COUNT']
    MAX_STEPS = DDPG_config['MAX_STEPS']
    if EXPLORE <= 1:
        EXPLORE = EPISODE_COUNT * MAX_STEPS * EXPLORE
    # SETUP ENDS HERE

    reward = 0
    done = False
    wise = False
    step = 0
    epsilon = 1
    indicator = 0

    #Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    

    actor = ActorNetwork(sess, state_dim, action_dim, DDPG_config)
    critic = CriticNetwork(sess, state_dim, action_dim, DDPG_config)
    buff = ReplayBuffer(BUFFER_SIZE)    #Create replay buffer

    ltm = ['a_h0', 'a_h1', 'a_V', 'c_w1', 'c_a1', 'c_h1', 'c_h3', 'c_V']
    layers_to_mind = {}
    L2 = {}

    for k in ltm:
        layers_to_mind[k] = 0
        L2[k] = 0

    vector_to_file(ltm, folder + 'weightsL2' + 'Log.csv', 'w')
    # 把ltm转化为String类型，写入weightsL2Log.csv文件中，作为行标题

    #Now load the weight
    try:
        actor.model.load_weights(folder + "actormodel.h5")
        critic.model.load_weights(folder + "criticmodel.h5")
        actor.target_model.load_weights(folder + "actormodel.h5")
        critic.target_model.load_weights(folder + "criticmodel.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("OMNeT++ Experiment Start.")
    # initial state of simulator
    s_t = env.reset()
    loss = 0
    for i in range(EPISODE_COUNT):

        print("Episode : " + str(i) + " Replay Buffer " + str(buff.count()))

        total_reward = 0
        for j in range(MAX_STEPS):
            print("step=:" + str(j))
            epsilon -= 1.0 / EXPLORE  #init esplion = 1
            a_t = np.zeros([1, action_dim]) # 1行,action_dim 列,元素为0
            noise_t = np.zeros([1, action_dim])
            print('action_dim',action_dim)
            a_t_original = actor.model.predict(s_t.reshape(1, s_t.shape[0])) # actornetwork.model
            # add action = (weight, bandwidth)
            print('a_t_original=\n',a_t_original)
            print('s_t=\n',s_t)
            
            # return type = numpy array
            # numpy.shape?? 多维=行数，一维=个数
            # numpy.reshape(i,j) Gives a new shape to an array without changing its data. 数据不变，改变了行数列数
            if train_indicator and epsilon > 0 and (step % 1000) // 100 != 9:  # init step = 0
                noise_t[0] = epsilon * ou.evolve() # evolove() 高斯分布之类的数学过程
                # train_indicator == 1 means train

            a = a_t_original[0]
            n = noise_t[0]
            # a_t[0] = np.where((a + n > 0) & (a + n < 1), a + n, a - n).clip(min=0, max=1)
            a_t[0] = a
            
            
            # if 0<(a+n)<1, return a+n, else return a-n 

            # numpy.clip(min,max) the elemnets who larger than max, will be replace by max, all elemnts between min and max
            
            # execute action
            s_t1, r_t, done = env.step(a_t[0],model, data_mean, data_std)  # call omnet
            print('after step self.env_T=\n')
            print(env.env_T)
            
            # state, action, reward, new_state,done

            buff.add(s_t, a_t[0], r_t, s_t1, done)      #Add replay buffer
            # s_t initial state 
            scale = lambda x: x
            #Do the batch update
            batch = buff.getBatch(BATCH_SIZE)
            states = scale(np.asarray([e[0] for e in batch]))
            actions = scale(np.asarray([e[1] for e in batch]))
            rewards = scale(np.asarray([e[2] for e in batch]))
            new_states = scale(np.asarray([e[3] for e in batch]))
            dones = np.asarray([e[4] for e in batch])

            y_t = np.zeros([len(batch), action_dim])
            target_q_values = critic.target_model.predict([new_states, actor.target_model.predict(new_states)])
            # ???
            # according to dones, init y_t
            for k in range(len(batch)):
                if dones[k]:
                    y_t[k] = rewards[k]
                else:
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

            # print('end batch for')

            if train_indicator and len(batch) >= BATCH_SIZE:
                loss = critic.model.train_on_batch([states, actions], y_t)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                # does this give an output like train_on_batch above? NO
                actor.train(states, grads)
                actor.target_train()
                critic.target_train()
                with open(folder + 'lossLog.csv', 'a') as file:
                    file.write(pretty(loss) + '\n')

            total_reward += r_t
            s_t = s_t1
            # print('begin layer')

            for layer in actor.model.layers + critic.model.layers:
                if layer.name in layers_to_mind.keys():
                    L2[layer.name] = np.linalg.norm(np.ravel(layer.get_weights()[0])-layers_to_mind[layer.name])
#                     vector_to_file(np.ravel(layer.get_weights()[0]), folder + 'weights_' + layer.name + 'Log.csv', 'a')
                    layers_to_mind[layer.name] = np.ravel(layer.get_weights()[0])
#             if max(L2.values()) <= 0.02:
#                 wise = True

            if train_indicator and len(batch) >= BATCH_SIZE:
                vector_to_file([L2[x] for x in ltm], folder + 'weightsL2' + 'Log.csv', 'a')

            vector_to_file(a_t_original[0], folder + 'actionLog.csv', 'a')
            vector_to_file(noise_t[0], folder + 'noiseLog.csv', 'a')

            if 'PRINT' in DDPG_config.keys() and DDPG_config['PRINT']:
                print("Episode", "%5d" % i, "Step", "%5d" % step, "Reward", "%.6f" % r_t)
                print("Epsilon", "%.6f" % max(epsilon, 0))

                att_ = np.split(a_t[0], ACTIVE_NODES)
                for _ in range(ACTIVE_NODES):
                    att_[_] = np.insert(att_[_], _, -1)
                att_ = np.concatenate(att_)
                print("Action\n", att_.reshape(ACTIVE_NODES, ACTIVE_NODES))
                print(max(L2, key=L2.get), pretty(max(L2.values())))

            step += 1
            if done or wise:
                break

        if np.mod((i+1), 2) == 0:   # writes at every 2nd episode
            if (train_indicator):
                actor.model.save_weights(folder + "actormodel.h5", overwrite=True)
                actor.model.save_weights(folder + "actormodel" + str(step) + ".h5")
                with open(folder + "actormodel.json", "w") as outfile:
                    outfile.write(actor.model.to_json(indent=4) + '\n')

                critic.model.save_weights(folder + "criticmodel.h5", overwrite=True)
                critic.model.save_weights(folder + "criticmodel" + str(step) + ".h5")
                with open(folder + "criticmodel.json", "w") as outfile:
                    outfile.write(critic.model.to_json(indent=4) + '\n')

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down
    print("Finish.")


if __name__ == "__main__":
    # QoE training
    model, data_mean, data_std = NN_training()
    # VANILLA
    if len(sys.argv) == 1:
        with open('DDPG.json') as jconfig:
            DDPG_config = json.load(jconfig)
        DDPG_config['EXPERIMENT'] = setup_exp()
        playGame(DDPG_config, model, data_mean, data_std, train_indicator=1)
    # PLAY
    elif len(sys.argv) == 3:
        # WATCH OUT: it appends to *Log.csv files
        if sys.argv[1] == 'play':
            with open(sys.argv[2] + '/' + 'DDPG.json') as jconfig:
                DDPG_config = json.load(jconfig)
            # here remove double slash at end if present
            experiment = sys.argv[2] if sys.argv[2][-1] == '/' else sys.argv[2] + '/'
            DDPG_config['EXPERIMENT'] = experiment
            playGame(DDPG_config, model, data_mean, data_std, train_indicator=0)
    # PLAY WITH FILE TRAFFIC
    elif len(sys.argv) == 4:
        # WATCH OUT: it appends to *Log.csv files
        if sys.argv[1] == 'play':
            with open(sys.argv[2] + '/' + 'DDPG.json') as jconfig:
                DDPG_config = json.load(jconfig)
            # here remove double slash at end if present
            experiment = sys.argv[2] if sys.argv[2][-1] == '/' else sys.argv[2] + '/'
            DDPG_config['EXPERIMENT'] = experiment
#             DDPG_config['EPISODE_COUNT'] = 1
#             DDPG_config['MAX_STEPS'] = 1
            if DDPG_config['TRAFFIC'] == 'DIR:':
                DDPG_config['TRAFFIC'] += sys.argv[3]
            playGame(DDPG_config, model, data_mean, data_std, train_indicator=0)
