#coding:utf-8
'''
Actor-Critic using TD-error as the advatage, Reinforcement learning.

所谓的TD-error，其实就是像DQN那样，输入状态s，输出值函数v(s)。
然后通过target net可以得到v(s')，所以TD-error = r + v(s') - v(s)

'''

import numpy as np
import tensorflow as tf
import gym
from Actor import Actor
from Critic import Critic

np.random.seed(1667)
tf.set_random_seed(1667)

# Superparameters
OUTPUT_GRAPH =False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 1000   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
LR_A = 0.001    # learning rate for actor
LR_C = 0.01     # learning rate for critic

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    env.seed(1667)
    env = env.unwrapped
    N_F = env.observation_space.shape[0]
    N_A = env.action_space.n

    sess = tf.Session()

    actor = Actor(sess = sess,
            n_features = N_F,
            n_actions = N_A,
            lr = LR_A)

    critic = Critic(sess = sess,
            n_features = N_F,
            gamma = GAMMA,
            lr = LR_C)

    sess.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        tf.summary.FileWriter('logs/', sess.graph)

    for epoch in range(MAX_EPISODE):
        s = env.reset()
        t = 0
        track_r = []

        while True:
            if RENDER: env.render()

            a = actor.choose_action(s)

            s_, r, done, info = env.step(a)

            if done: r = -20

            track_r.append(r)

            td_error = critic.learn(s, r, s_)

            actor.learn(s, a, td_error)

            s = s_
            t +=1

            if done or t>=MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                if 'running_reward' not in globals():
                    running_reward = ep_rs_sum
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", epoch, "  reward:", int(ep_rs_sum))
                break

