#coding:utf-8

class config(object):
    def __init__(self):
        self.env_name = 'Pendulum-v0'  # 'CartPole-v0'
        self.max_episodes = 500
        self.max_ep_steps = 200
        self.lr_a = 0.001  # learning rate for actor network
        self.lr_c = 0.001  # learing rate for critic network
        self.memory_size = 30000 
        self.batch_size = 32
        self.is_render = True
        self.gamma = 0.9 
        self.is_output_graph = False
        self.TAU = 0.01
