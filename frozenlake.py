import gym
import numpy as np
from collections import deque


def main():

    # Initialization of parameters
    env = gym.make('FrozenLake-v0', is_slippery=True)
    env.seed(0)
    np.random.seed(0)
    epsilon = 0.5
    epsilon_decay = 0.9
    gamma = 0.9
    alpha = 0.99
    render = False
    verbose = True
    ep_num = 0
    total_rewards = deque([0.0], maxlen=100)

    # Initialize Q-table
    # Set terminal states to zero
    Q = np.random.random((env.observation_space.n, env.action_space.n))
    Q[5] = [0., 0., 0., 0.]
    Q[7] = [0., 0., 0., 0.]
    Q[11] = [0., 0., 0., 0.]
    Q[12] = [0., 0., 0., 0.]
    Q[15] = [0., 0., 0., 0.]

    # Actions:
    # 0: Left
    # 1: Down
    # 2: Right
    # 3: Up

    # Use an epsilon-greedy strategy to choose actions
    def choose_action(_state):
        if np.random.uniform(0, 1) < epsilon:
            chosen_action = np.random.randint(env.action_space.n)
        else:
            chosen_action = np.argmax(Q[_state, :])
        return chosen_action

    # Update the Q-table
    def learn(_state, _new_state, _reward, _action):
        predict = Q[_state, _action]
        target = _reward + gamma * np.max(Q[_new_state, :])
        Q[_state, _action] = (1 - alpha) * predict + alpha * target

    # The problem is solved when the average score over the last 100 episodes is 0.78 or greater
    # Begin episode
    while np.mean(total_rewards) < 0.78:
        state = env.reset()
        total_reward = 0
        if render:
            env.render()

        # Begin timestep
        while True:
            action = choose_action(state)

            # Decay epsilon
            epsilon = max(epsilon_decay * epsilon, 0.00001)

            # Take the chosen action
            new_state, reward, done, _ = env.step(action)

            # Update the Q-table
            learn(state, new_state, reward, action)

            state = new_state
            total_reward += reward

            if render:
                env.render()
            if done:
                break

        total_rewards.append(total_reward)
        ep_num += 1
        if verbose:
            print("Episode:\t" + str(ep_num) + "\tEpsilon:\t" + str(epsilon) + "\tReward:\t" + str(np.mean(total_rewards)))


if __name__ == "__main__":
    main()
