import gym
import argparse
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class policyGradient:
    def __init__(self, 
                n_actions,
                n_features,
                lr = 0.01,
                gamma = 0.95,
                output_graph = False,
                ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = gamma
        self.output_graph = output_graph
        
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        
        self._build_net() #建立policy神经网络
        
        self.sess = tf.Session()
        
        if self.output_graph:
            tf.summary.FileWriter('../logs/', self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())
        
        
    
    def _build_net(self):
        with tf.name_scope('inputs'):
            self.tf_obs = tf.placeholder(tf.float32, 
                                         [None, self.n_features], 
                                         name="observations")  # 接收 observation
            
            # 接收我们在这个回合中选过的 actions
            self.tf_acts = tf.placeholder(tf.int32, [None, ], 
                                          name="actions_num")   
            
            # 接收每个 state-action 所对应的 value (通过 reward 计算)
            self.tf_vt = tf.placeholder(tf.float32, [None, ], 
                                        name="actions_value") 
            
        with tf.name_scope('net'):
            fc1 = tf.layers.dense(inputs = self.tf_obs,
                                 units = 10, #输出神经元个数
                                 activation = tf.nn.tanh,
                                 kernel_initializer = tf.random_normal_initializer(0,0.3),
                                 bias_initializer = tf.constant_initializer(0.1),
                                 name = 'fc1'
                                 )
            all_act = tf.layers.dense(inputs = fc1,
                                      units = self.n_actions,
                                      activation = None, 
                                      kernel_initializer = tf.random_normal_initializer(0,0.3),
                                      bias_initializer = tf.constant_initializer(0.1),
                                      name = 'fc2',
                                     )
            
        self.all_act_prob = tf.nn.softmax(all_act, name='act_prob')#sofrmax计算每个动作被选到的概率
        
        with tf.name_scope('loss'):
            neg_loss_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = all_act,
                                                                          labels = self.tf_acts)
            loss = tf.reduce_mean(neg_loss_prob*self.tf_vt)## (vt = 本reward + 衰减的未来reward) 引导参数的梯度下降
            
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            
            
            
            
    def choose_action(self, observation):
        #将observation输入网络，网络返回关于所有action的概率分布，
        #然后我们可以根据概率分布来选择动作，具有一定的随机性性
        prob_weights = self.sess.run(self.all_act_prob, 
                                     feed_dict = {self.tf_obs: observation[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action 
    
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    
    def learn(self):
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        
        feed_dict = {self.tf_obs: np.vstack(self.ep_obs), #shape = [None, n_obs]
                     self.tf_acts: np.array(self.ep_as), #shape = [None, ]
                     self.tf_vt: discounted_ep_rs_norm, #shape = [None, ]
                     }
        
        self.sess.run(self.train_op, feed_dict = feed_dict)
    
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        return discounted_ep_rs_norm #返回这一回合的state-action value
        
        
    def _discount_and_norm_rewards(self):
        """衰减回合的reward？？？？"""
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add*self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        
        return discounted_ep_rs

