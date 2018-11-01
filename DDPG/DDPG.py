"""
Deep Deterministic Policy Gradient (DDPG)
"""
import tensorflow as tf
import numpy as np

class DDPG(object):
    def __init__(self, config, a_dim, s_dim):
        """
        @config: A object that saves some hyperparameters 
        @a_dim: The dimension of action feature
        @s_dim: The dimension of state feature
        """
        self.config = config
        self.memory = np.zeros((self.config.memory_size, s_dim * 2 + a_dim + 1), dtype = np.float32)

        self.pointer = 0

        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_actor(self.S) # build the policy gradient network(actor)
        q = self._build_critic(self.S, self.a) # build the Q network (critic), note that q is built by actor and state

        actor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Actor')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Critic')

        ema = tf.train.ExponentialMovingAverage(decay=1-self.config.TAU)  # soft replacement for trainable parameters

