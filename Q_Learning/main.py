from gridworld import *
from agent import *

import time

max_episode = 100000

def update(agent, env):
    for episode in range(max_episode):
        s = env.reset()
        env.render()
        time_in_episode = 0
        is_done = False
        while not is_done:
            

            a = agent.choose_action(s, episode, True)

            s_, r, is_done, info = env.step(a)
            

            #fresh env for display
            env.render()

            agent.learn(s, a, r, s_)

            s = s_
            time_in_episode += 1

        print("Episode {0} takes {1} steps.".format(
                episode+1, time_in_episode))

    print('game over!')
    env.destroy()


if __name__ == '__main__':
    env = SimpleGridWorld()

    agent = Agent(env=env,
            gamma = 0.9,
            epsilon = 0.0,
            alpha = 0.1)

    update(agent, env)

