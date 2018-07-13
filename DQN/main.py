#coding:utf-8
from DQN import DQN
import gym
import argparse
import sys

def CartPole_v0(args):
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    #
    # print(env.action_space) # 查看这个环境中可用的 action 有多少个
    # print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
    # print(env.observation_space.high)   # 查看 observation 最高取值
    # print(env.observation_space.low)    # 查看 observation 最低取值

    agent = DQN(n_actions = env.action_space.n, 
                n_features = env.observation_space.shape[0],
                lr = args.lr,
                gamma = args.gamma, 
                epsilon = args.epsilon,
                replace_target_iter = args.update_when,
                memory_size = args.memory_size,
                batch_size = args.batch_size,
                epsilon_decreament = 0.0008,
                drop_out = args.drop_out)

    total_step = 0
    for epoch in range(1, args.epoches+1):
        observation = env.reset()
        reward_every_epoch = 0

        while True:
            env.render() #刷新环境

            action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            # x: 小车的水平位置, x==0就是在中间位置。
            # theta: 棒与垂直地面的线的夹角，角度越大，棒就越倾向地面，越不稳定。 
            x, x_dot, theta, theta_dot = observation_
            r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5

            reward = r1 + r2

            agent.store_transition(observation, action, reward, observation_)

            if total_step > 1000:
                agent.learn()

            reward_every_epoch += reward
            if done:
                print('episode: ', epoch,
                      '| reward_every_epoch: ', round(reward_every_epoch, 2),
                      '| epsilon: ', round(agent.epsilon,2))
                break

            observation = observation_
            total_step += 1
            
            
    agent.plot_cost()
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v:v.lower() == 'true')
    
    parser.add_argument(
        '--memory_size',
        type = int,
        default = 2000,
        help='the size of the memory of the agent.(default = 2000)')
    
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

    CartPole_v0(args)


