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


    ddpg = DDPG(config, env)

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

            ddpg.store_transition(s, a, r, s_)

            ep_reward += r

            if ddpg.memory_full:
                ddpg.learn()

            s = s_
            if done: 
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (episode, 'done', ep_reward, step))
                break

            if step == config.max_episodes:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (episode, '---' , ep_reward, step))
                break
    # ddpg.save()

