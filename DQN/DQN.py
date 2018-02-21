from RL_brain import DeepQNetwork
import gym



env = gym.make('CartPole-v0')
env = env.unwrapped
#
# print(env.action_space) # 查看这个环境中可用的 action 有多少个
# print(env.observation_space)    # 查看这个环境中可用的 state 的 observation 有多少个
# print(env.observation_space.high)   # 查看 observation 最高取值
# print(env.observation_space.low)    # 查看 observation 最低取值

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01,
                  e_greedy=0.9,
                  replace_target_iter=100,
                  memory_size=2000,
                  e_greedy_increment=0.0008,
                  )

total_step = 0  #record step number

for i_episode in range(100):
    observation = env.reset()
    ep_r = 0

    while True:
        env.render()  #刷新环境

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # x 是车的水平位移, 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子离垂直的角度, 角度越大, 越不垂直. 所以 r2 是棒越垂直, 分越高
        x, x_dot, theta, theta_dot = observation_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians -0.5

        reward = r1 + r2

        RL.store_transition(observation, action, reward, observation_)

        if total_step > 1000:
            RL.learn()

        ep_r += reward
        if done:
            print('episode:',i_episode,
                  'ep_r:', round(ep_r, 2),
                  'epsilon:', round(RL.epsilon,2)
                  )
            break

        observation = observation_
        total_step += 1


RL.plot_cost()
