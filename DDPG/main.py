#coding:utf-8
import tensorflow as tf
import sys
import gym
from config import config
from DDPG import DDPG


if __name__=='__main__':
    config = config()
    env = gym.make(config.env_name)
    env = env.unwrapped
    env.seed(1667)

    a_dim = env.action_space.shape[0]
    s_dim = env.observation_space.shape[0]
    a_bound = env.action_space.high

    ddpg = DDPG(config, a_dim, s_dim, a_bound)

    for episode in range(config.max_episodes):
        s = env.reset()
        ep_reward = 0

        for step in range(config.max_ep_steps):
            # judge to refresh the window 
            if config.is_render and ddpg.memory_full:
                env.render()

            a = ddpg.choose_action(s)
            # print('the value of action: ', a)
            s_, r, done, info = env.step(a)

            ddpg.store_transition(s, a, r/10, s_)

            ep_reward += r

            if ddpg.memory_full:
                ddpg.learn()

            s = s_
            if done: 
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (episode, 'done', ep_reward, step))
                break

            if step == config.max_ep_steps-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (episode, '---' , ep_reward, step))
                break
    # ddpg.save()

