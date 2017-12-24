from gridworld import *
from agent import *

import time

max_episode = 100000




if __name__ == '__main__':
    env = SimpleGridWorld()

    agent = Agent(env=env,
            gamma = 0.9,
            epsilon = 0.0,
            alpha = 0.1,
            max_episode = max_episode)

    #update(agent, env)

