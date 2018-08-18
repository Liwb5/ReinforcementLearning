#coding:utf-8
import numpy as np
import tensorflow as tf
import gym


class Critic(object):
    def __init__(self, sess,
            n_features,
            lr = 0.01,
            gamma = 0.9):
        self.sess = sess
        self.lr = lr
        self.gamma = gamma
        self.n_features = n_features
        self._build_net()

    def _build_net(self):
        self.s = tf.placeholder(tf.float32, [1, self.n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')


	with tf.variable_scope('Critic'):
	    L1 = tf.layers.dense(
		inputs=self.s,
		units=20,  # number of hidden units
		activation=tf.nn.relu,  # None
		# have to be linear to make sure the convergence of actor.
		# But linear approximator seems hardly learns the correct Q.
		kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
		bias_initializer=tf.constant_initializer(0.1),  # biases
		name='L1'
	    )

	    self.v = tf.layers.dense(
		inputs=L1,
		units=1,  # output units
		activation=None,
		kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
		bias_initializer=tf.constant_initializer(0.1),  # biases
		name='V'
	    )

	with tf.variable_scope('squared_TD_error'):
	    self.td_error = self.r + self.gamma * self.v_ - self.v
	    self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
	with tf.variable_scope('train'):
	    self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r})
        return td_error
 









