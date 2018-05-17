from doubleDQN import DDQN
import gym
import argparse
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

ACTION_SPACE = 11 #将原本的连续动作分为11个离散动作
MEMORY_SIZE = 3000


def train(agent, env):
    total_step = 0
    observation = env.reset()

    while True:
        if total_step - MEMORY_SIZE > 8000: env.render()
            
        action = agent.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # 在 [-2 ~ 2] 内离散化动作
        
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10 

        agent.store_transition(observation, action, reward, observation_)

        if total_step > MEMORY_SIZE:
            agent.learn()
            
        if total_step - MEMORY_SIZE > 20000:
            break # stop game
              
        observation = observation_
        total_step += 1
    return agent.q #返回所有的动作Q值


def Pendulum_v0(args):
    env = gym.make('Pendulum-v0')
    env = env.unwrapped
    env.seed(1)
    #
    # print(env.action_space) # 查看这个环境中可用的 action 有多少个
    # print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
    # print(env.observation_space.high)   # 查看 observation 最高取值
    # print(env.observation_space.low)    # 查看 observation 最低取值

    sess = tf.Session()
    with tf.variable_scope('Natural_DQN'):
        agent2 = DDQN(n_actions = ACTION_SPACE, 
                    n_features = 3,
                    lr = args.lr,
                    gamma = args.gamma, 
                    epsilon = args.epsilon,
                    replace_target_iter = args.update_when,
                    memory_size = MEMORY_SIZE,
                    batch_size = args.batch_size,
                    epsilon_decreament = 0.0008,
                    drop_out = args.drop_out,
                    double_q = False, #not using double DQN
                    sess = sess,
                    )
        
    with tf.variable_scope('Double_DQN'):
        agent3 = DDQN(n_actions = ACTION_SPACE, 
                    n_features = 3,
                    lr = args.lr,
                    gamma = args.gamma, 
                    epsilon = args.epsilon,
                    replace_target_iter = args.update_when,
                    memory_size = MEMORY_SIZE,
                    batch_size = args.batch_size,
                    epsilon_decreament = 0.0008,
                    drop_out = args.drop_out,
                    double_q = True, #using double DQN
                    sess = sess,
                    )
        
    sess.run(tf.global_variables_initializer())

    q_natural = train(agent2, env)
    q_double = train(agent3, env)
    
    # 出对比图
    plt.plot(np.array(q_natural), c='r', label='natural')
    plt.plot(np.array(q_double), c='b', label='double')
    plt.legend(loc='best')
    plt.ylabel('Q eval')
    plt.xlabel('training steps')
    plt.grid()
    plt.show()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v:v.lower() == 'true')
    
    
    parser.add_argument(
        '--gamma',
        type = float,
        default = 0.9, 
        help='the discount factor of the reward.(default = 0.9)')
    
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.01,
        help = 'the learning rate of the model(default = 0.01)')
    
    parser.add_argument(
        '--epoches',
        type = int,
        default = 100,
        help = 'the max number of epoch(default = 100)')
    
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 32,
        help = 'the batch size of train data(default = 128)')
    
    parser.add_argument(
        '--drop_out',
        type = float,
        default = 0.5,
        help = 'the probability of drop out(default = 0.5) ')

    parser.add_argument(
        '--epsilon',
        type = float,
        default = 0.9,
        help = 'the e_greedy probability.(default = 0.9) ')
    
    #每隔多少步就将eval net的参数更新到target net中
    parser.add_argument(
        '--update_when',
        type = int,
        default = 100,
        help = 'update the params in eval net to target net.(default = 500)')
    
    args, unparsed = parser.parse_known_args()  
    print(args)

    Pendulum_v0(args)


