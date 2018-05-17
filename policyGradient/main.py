from policyGradient import policyGradient
import gym
import argparse
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


RENDER = False 
DISPLAY_REWARD_THRESHOLD = 400 #当回合总的reward大于阈值时显示模拟窗口
learning_rate = 0.02
gamma = 0.99
MAX_EPOCH = 3000


def train(agent, env, RENDER=RENDER):
    #Q learning， DQN都是每走一步更新一次，这里是每回合更新一次
    for epoch in range(MAX_EPOCH):
        observation = env.reset()
        
        while True:
            if RENDER: 
                env.render()
            
            action = agent.choose_action(observation)
            
            observation_, reward, done, info = env.step(action)
            
            agent.store_transition(observation, action, reward)#不需要保存下一个状态了
            
            if done:
                ep_rs_sum = sum(agent.ep_rs)#计算整个回合得到的reward
                
                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward*0.99 + ep_rs_sum*0.01
                
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
                
                print('episode: ', epoch, '| reward: ', int(running_reward))
                
                vt = agent.learn() #学习，输出vt
                
                if epoch == 0:
                    plt.plot(vt)    # plot 这个回合的 vt
                    plt.xlabel('episode steps')
                    plt.ylabel('normalized state-action value')
                    plt.show()
                break
            observation = observation_
            

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env = env.unwrapped  #取消限制
    env.seed(1)
    
    agent = policyGradient(
        n_actions = env.action_space.n,
        n_features = env.observation_space.shape[0],
        lr = learning_rate,
        gamma = gamma,
        output_graph = False,
        )

    train(agent, env)
