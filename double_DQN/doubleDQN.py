import tensorflow as tf
import pandas as pd
import numpy as np

np.random.seed(1)
tf.set_random_seed(1)

class DDQN:
    def __init__(self, 
                n_actions, #动作空间，即有多少个动作
                #每个状态用一个向量表示，n_features就是指向量的长度.即agent观察到的特征
                n_features, 
                lr = 0.01,
                gamma = 0.9, #奖赏的折扣因子
                epsilon = 0.9,#随机探索的概率
                #DQN有两个network，这个参数表示每隔多少步将eval network的参数更新到target network中去。
                replace_target_iter = 300, 
                memory_size = 500, #记忆库容量的大小
                batch_size = 32,
                epsilon_decreament = None,#随着学习的进行，是否逐渐减少探索的概率
                drop_out = 0.5,
                output_graph = False,
                double_q = True,
                sess = None,
                ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_decreament = epsilon_decreament
        self.drop_out =drop_out
        self.double_q = double_q
        
        #self.epsilon_max = self.epsilon
        #self.epsilon = 0 if epsilon_decreament is not None else self.epsilon
        
        # total learning step
        self.learn_step_counter = 0 #用于记录学习了多少步
        #每一行包含(s,a,r,s_)，s和s_是一个长度为n_features的向量，所以乘以2.
        #加2则是因为r和a各占一个。
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
        
        self._build_net()
        
        #将eval_net的参数更新到target_net中，需要sess.run()才会执行。
        eval_params = tf.get_collection('eval_net_params')
        target_params = tf.get_collection('target_net_params')
        self.replace_params_operation = [tf.assign(t, e) for t, e in zip(target_params, eval_params)]
        
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
            
        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logsDQN/", self.sess.graph)
        
        self.cost_his = []#记录loss值，最后画出loss曲线
    
    def _build_net(self):
        # ------------------ build evaluate_net ------------------#
        # s is the input of the network
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')
        # q_target is regarded as the label of the network, 
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')
        
        self.drop_prob = tf.placeholder(tf.float32, name='drop_prob')
        
        num_L1 = 10 #神经网络第一层的输出神经元个数
        
        #build the network
        with tf.variable_scope('eval_net'):
            
            # evaluate net的参数将会放在下面这两个集合下，第一个是自定义的。第二个是默认的
            collections_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
        
            eval_L1 = self._add_layer(inputs = self.s, 
                            var_scope = 'L1', 
                            in_size = self.n_features, 
                            out_size = num_L1, 
                            collections_name = collections_name,
                            acti_func = tf.nn.relu,
                            isDropout = False)
            
            self.eval_L2 = self._add_layer(inputs = eval_L1, 
                            var_scope = 'L2', 
                            in_size = num_L1, 
                            out_size = self.n_actions, 
                            collections_name = collections_name,
                            acti_func = None,
                            isDropout = False)
            
        #define the loss function
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.eval_L2))
            
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        
        
        # ------------------ build target_net ------------------#
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')
        
        with tf.variable_scope('target_net'):
            
            # evaluate net的参数将会放在下面这两个集合下，第一个是自定义的。第二个是默认的
            collections_name = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            
            tar_L1 = self._add_layer(inputs = self.s_, 
                            var_scope = 'L1', 
                            in_size = self.n_features, 
                            out_size = num_L1, 
                            collections_name = collections_name,
                            acti_func = tf.nn.relu,
                            isDropout = False)
            
            self.tar_L2 = self._add_layer(inputs = tar_L1, 
                            var_scope = 'L2', 
                            in_size = num_L1, 
                            out_size = self.n_actions, 
                            collections_name = collections_name,
                            acti_func = None,
                            isDropout = False)
        
        
    def _add_layer(self, inputs, var_scope, in_size, out_size, 
                   collections_name=None, acti_func = None, isDropout=False):
        
        with tf.variable_scope(var_scope):
            W = tf.get_variable('weights', [in_size, out_size], 
                                initializer=tf.random_normal_initializer(0., 0.3),
                                collections = collections_name)
            
            b = tf.get_variable('bias', [1, out_size],
                                initializer = tf.constant_initializer(0.1),
                                collections = collections_name)
            
            if acti_func is None:
                y = tf.matmul(inputs, W) + b
            else:
                y = acti_func(tf.matmul(inputs, W) + b)
        
            if isDropout:
                y = tf.nn.dropout(y, keep_prob=self.drop_prob)
            return y
            
        
        
    def store_transition(self, s, a, r, s_):
        """保存一次动作后的当前状态s，动作a，奖赏r，下一个状态s_
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
        
    def choose_action(self, obsevation):
        """输入一个obsevation（s），让神经网络输出一个action。
        以前没有神经网络的时候是直接找Q值最大对应的action，
        现在的Q值是由神经网络产生的。
        
        obsevation作为eval_net的输入，eval_net是用来预测的。target_net是作为label的。
        
        return：
            action：int型的数据
        """
        observation = obsevation[np.newaxis, :]
        actions_value = self.sess.run(self.eval_L2, feed_dict={self.s: observation})
        action = np.argmax(actions_value)
        
        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)
        
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
            
        return action
    
    
    def learn(self):
        """"""
        #检查是否要将eval_net的参数更新到target_net
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_params_operation)
            print('\n target_params_replaced \n')
            
        #说明记忆库已经填满了，可以随机从中采样
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            
        else:
            sample_index = np.random.choice(self.memory_counter, size = self.batch_size)
        
        batch_memory = self.memory[sample_index, :]
        
        feed_dict = {self.s_: batch_memory[:,-self.n_features:],
                     self.s: batch_memory[:, :self.n_features]}
        #我们要把q_next当成是最优的，利用它来更新q_target
        q_next, q_eval = self.sess.run([self.tar_L2, self.eval_L2],
                                         feed_dict = feed_dict)
        
        #next obsevation 用eval net得到Q，从中选出要执行的动作a'
        q_eval4next = self.sess.run(self.eval_L2, 
                                    feed_dict = {self.s: batch_memory[:,-self.n_features:]})
        
        q_target = q_eval.copy()#因为选择动作的时候只会选择一个，所以要保证其他的动作不会影响到loss
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]
        
        if self.double_q:
            #利用eval net来得到Q，选择a'，再选择q_next中的Q值。而不是直接选择q_next的最大值。
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)# natural DQN
            
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        
        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]

        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]

        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]

        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """        
        
        #经过上面的计算q_target才被认为是label
        #接下来再进行训练
        feed_dict = {self.s: batch_memory[:, :self.n_features],
                     self.q_target: q_target}
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict = feed_dict)
        
        #保存loss值，方便画图
        self.cost_his.append(self.cost)
        
        self.learn_step_counter += 1
        
        #这里需要考虑epsilon随时间变小
        if self.epsilon > 0.1:
            self.epsilon = self.epsilon - self.epsilon_decreament
            
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()        
        