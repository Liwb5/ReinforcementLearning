

class config(object):
    def __init__(self):
        self.max_episoders = 200
        self.max_ep_steps = 200
        self.lr_a = 0.001  # learning rate for actor network
        self.lr_c = 0.002  # learing rate for critic network
        self.memory_size = 10000 
        self.TAU = 0.01  # for soft replacment for trainable prameters
        self.batch_size = 32
        # self.render = False
        
