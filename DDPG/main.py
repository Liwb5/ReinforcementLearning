import tensorflow as tf
import sys
import gym

ENV_NAME = 'CartPole-v0'

if __name__=='__main__':
    env = gym.make(ENV_NAME)
    env = env.unwrapped
    env.seed(1667)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
