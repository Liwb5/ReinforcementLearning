"""
Deep Deterministic Policy Gradient (DDPG)
"""
import tensorflow as tf
import numpy as np

class DDPG(object):
    def __init__(self, config, env):
        """
        @config: A object that saves some hyperparameters 
        @env: The environment object
        """
        self.config = config
        
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]
        self.a_bound = env.action_space.high

        self.memory = np.zeros((self.config.memory_size, self.s_dim * 2 + self.a_dim + 1), dtype = np.float32)

        self.pointer = 0 # for indicate the memory
        self.memory_full = False

        self.sess = tf.Session()

        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_actor(self.S, scope='eval', trainable=True) # build the eval policy gradient network(actor)
            self.a_ = self._build_actor(self.S_, scope='target', trainable=False) # build the target policy gradient network(actor)

        with tf.variable_scope('Critic'):
            self.q = self._build_critic(self.S, self.a, scope='eval', trainable=True) # build the Q network (critic), note that q is built by actor and state
            self.q_ = self._build_critic(self.S_, self.a_, scope='target', trainable=False) 

        self.ae_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'Critic/target')


        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - self.config.TAU) * ta + self.config.TAU * ea), 
                            tf.assign(tc, (1 - self.config.TAU) * tc + self.config.TAU * ec)]
                            for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + self.config.gamma* self.q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=self.q)
        self.ctrain = tf.train.AdamOptimizer(self.config.lr_c).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(self.q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(self.config.lr_a).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())


    def choose_action(self, s):
        s = s[np.newaxis, :]
        res = self.sess.run(self.a, feed_dict = {self.S: s})
        return res[0]

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.config.memory_size
        self.memory[index, :] = transition
        self.pointer += 1
        if self.memory_full == False and self.pointer > self.config.memory_size:
            self.memory_full = True

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(self.config.memory_size, size=self.config.batch_size)
        batch_samples = self.memory[indices, :]
        batch_s = batch_samples[:, :self.s_dim]
        batch_a = batch_samples[:, self.s_dim:(self.s_dim + self.a_dim)]
        batch_r = batch_samples[:, -self.s_dim-1: -self.s_dim]
        batch_s_ = batch_samples[:, -self.s_dim:]

        self.sess.run(self.atrain, feed_dict = {self.S: batch_s})
        self.sess.run(self.ctrain, feed_dict = {self.S: batch_s, self.a: batch_a, 
                                                self.R: batch_r, self.S_:batch_s_})
        
    def _build_critic(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 300
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)


    def _build_actor(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 300, activation=tf.nn.relu, name='L1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')
'''
class Actor(object):
    def __init__(self, sess, config, env, S, S_):
        self.sess = sess
        self.config = config
        self.a_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high

        with tf.variable_scope('Actor'):
            self.a = self._build_net(S, scope='eval_net', trainable= True)

            self.a_ = self._build_net(S_, scope='target_net', trainable = False)

        self.e_parms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/eval_net')
        self.t_parms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor/target_net')

        # choose which way to update parameters
        if self.config.replacement['name'] == 'ema':
            self.ema_replace_params = [tf.assign(t, (1-self.config.replacement['tau']) * t + self.config.replacement['tau'] * e)
                    for t, e in zip(self.t_parms, self.e_parms)]
        else:
            self.t_replace_counter = 0
            self.hard_replace_params = [tf.assign(t, e) for t, e in zip(self.t_parms, self.e_parms)]
        

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            L1 = tf.layers.dense(s, 30, activation = tf.nn.relu,
                    kernel_initializer = init_w,
                    bias_initializer = init_b, 
                    name = 'L1',
                    trainable = trainable)
            
            with tf.variable_scope('a'):
                actions = tf.layers.dense(L1, self.a_dim, activation = tf.nn.tanh,
                    kernel_initializer = init_w,
                    bias_initializer = init_b, 
                    name = 'a',
                    trainable = trainable)

                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')

        return scaled_a

    def learn(self, s): #batch update
        self.sess.run(self.train_op, feed_dict={S:s})

        if self.config.replacement['name'] == 'ema':
            self.sess.run(self.ema_replace_params)
        else:
            if self.t_replace_counter % self.config.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace_params)
            self.t_replace_counter += 1


    def choose_action(self, s):
        s = s[np.newaxis, :] # single state

        return self.sess.run(self.a, feed_dict = {S:s})[0] # single action

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            self.policy_grads = tf.gradient(ys=self.a, xs=self.e_prams, grad_ys=a_grads) # grad_ys 是权重

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.config.lr_a) # 负号才是最小化
            self.train_op = opt.apply_gradient(zip(self.policy_grads, self.e_prams))


class Critic(object):
    def __init__(self, sess, config, env, S, S_, a, a_):
        self.sess = sess
        self.config = config
        self.s_dim = env.observation_space.shape[0]
        a_dim = env.action_space.shape[0]

        with tf.variable_scope('Critic'):
            self.q = self._build_net(S, a, scope='eval_net', trainable=True)
            self.q_ = self._build_net(S_, a_, scope='target_net', trainable=False)


        self.e_parms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/eval_net')
        self.t_parms = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic/target_net')

        # choose which way to update parameters
        if self.config.replacement['name'] == 'ema':
            self.ema_replace_params = [tf.assign(t, (1-self.config.replacement['tau']) * t + self.config.replacement['tau'] * e)
                    for t, e in zip(self.t_parms, self.e_parms)]
        else:
            self.t_replace_counter = 0
            self.hard_replace_params = [tf.assign(t, e) for t, e in zip(self.t_parms, self.e_parms)]

    def _build_net(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q
'''
