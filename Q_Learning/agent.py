import numpy as np
import pandas as pd
import time
from random import random
from gym import Env
import gym
from gridworld import *

'''
gridworld类构建了一些简单的格子世界。
在格子世界中，每个格子相当于一个状态，状态的描述是按照行来排序的，第x行第y个的状态名为
x* 列数 + y （注意第一行x为0）

env.reset()函数重置了格子世界，agent每次走完一个episode都要reset()

env.action_space 是动作空间，一般格子世界中，每个状态下的动作有四个，
分别是左右上下。用0,1,2,3表示


env.step(a) 根据选择的动作进行状态转移 a = 0, 1, 2, 3 分别代表左 右 上 下
所以建立Q table的时候，列要用0, 1, 2, 3来表示
 
'''


class Agent:
    def __init__(self, env, gamma, epsilon, alpha, max_episode):
        self.env = env
        
        self.QTable = pd.DataFrame(columns = list(range(self.env.action_space.n)))
        self.gamma = gamma
        self.epsilon = epsilon
        self.alpha = alpha
        self.state = None
        self.max_episode = max_episode
        self._initAgent()

    def _initAgent(self):
        #gridworld文件中也没有定义reset，是gym自带的，可以将格子世界恢复到初始状态，agent经过每个episode之后都要reset一下。
        self.state = self.env.reset()

    def check_state_exist(self, state):
        if state not in self.QTable.index:
            #append new state to Q table
            self.QTable = self.QTable.append(
                pd.Series(
                    [0]*self.env.action_space.n,
                    index = self.QTable.columns,
                    name = state,
                    )
                )

    def _get_Q(self, s, a):
        return self.QTable.ix[s, a]

    def _set_Q(self, s, a, value):
        self.QTable.ix[s, a] = value


    def choose_action(self, state, episode_num, use_epsilon):
        self.check_state_exist(state)

        self.epsilon = 1.00 / (episode_num + 1)

        #指数型下降
        #self.epsilon = math.exp(-(episode_num+80)/80)

        #e-greedy algorithm
        if use_epsilon and np.random.uniform() < self.epsilon:
            #random choose an action
            action = self.env.action_space.sample()
        else:
            #choose best action
            state_action = self.QTable.ix[state, :]#获得当前状态state对应的所有动作的Q(state, a)值
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.argmax()

        return action


    def update(self, s, a, r, s_, a_):
        self.check_state_exist(s_)

        old_Q = self._get_Q(s, a)

        prime_Q = self._get_Q(s_, a_)

        Qtarget = r + self.gamma * prime_Q
            
        new_Q = old_Q + self.alpha * (Qtarget - old_Q)
        self._set_Q(s, a, new_Q) #更新Q值

        ##for set v(s) to gridworld
        value = self.QTable.ix[s, :].max() #获得当前状态state对应的所有动作的Q(state, a)值的最大值
        self.env.set_Q_value(s, value)


    def learning(self):
        for episode in range(self.max_episode):
            s = self.env.reset()
            self.env.COLOR = True
            self.env.render()
            self.env.COLOR = False
            time_in_episode = 0
            is_done = False
            while not is_done:
                
                a = self.choose_action(s, episode, use_epsilon=True)

                s_, r, is_done, info = self.env.step(a)
                
                #use_epsilon=False 表示是用greedy而不是e-greedy。这是Q learning的做法
                a_ = self.choose_action(s_, episode, use_epsilon=False)

                #fresh env for display
                self.env.render()

                self.update(s, a, r, s_, a_)

                s = s_
                time_in_episode += 1

            print("Episode {0} takes {1} steps. epsilon is {2:.3f}".format(
                    episode+1, time_in_episode, self.epsilon))

        print('game over!')
        self.env.destroy()

if __name__ == "__main__":

    max_episode = 1000000

    env = SimpleGridWorld()
    #env = CliffWalk()
    #env = WindyGridWorld()

    agent = Agent(env=env,
            gamma = 0.9,
            epsilon = 0.0,
            alpha = 0.1,
            max_episode = max_episode)

    agent.learning()


