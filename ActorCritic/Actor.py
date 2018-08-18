#coding:utf-8
import numpy as np
import tensorflow as tf
import gym


class Actor(object):
    def __init__(self, sess, 
            n_features,  #the number of features of the state
            n_actions,
            lr = 0.001):

        self.sess = sess
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr

        self._build_net()

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], 'state')
        self.a = tf.placeholder(tf.int32, None, 'act')
        self.td_error = tf.placeholder(tf.float32, None, 'td_error')
        
        with tf.variable_scope('Actor'):
            L1 = tf.layers.dense(
                    inputs = self.s, # input the state
                    units = 20, #number of hidden units
                    activation = tf.nn.relu,
                    kernel_initializer = tf.random_normal_initializer(0, 1),
                    bias_initializer = tf.constant_initializer(0.1),
                    name = 'L1')

            self.acts_prob = tf.layers.dense(
                    inputs = L1, # intput last layer's output
                    units = self.n_actions,
                    activation = tf.nn.softmax, 
                    kernel_initializer = tf.random_normal_initializer(0, 1),
                    bias_initializer = tf.constant_initializer(0.1),
                    name = 'acts_prob')

        with tf.variable_scope('exp_v'): # define loss
            log_prob = tf.log(self.acts_prob[0, self.a]) #self.a is the action that the agent really take.  
            self.exp_v = tf.reduce_mean(log_prob * self.td_error) # we want to make both log_prob and td_error to be large. This means the action taken is good

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v) #maxmimze(exp_v) == minimize(-exp_v)

    def learn(self, s, #current state 
            a, #current action that agent takes
            td_error #TD-error that calculated by Critic
            ):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td_error}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict) # exp_v is the loss
        return exp_v

    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.s: s}) #get probabilities of all actions
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())









