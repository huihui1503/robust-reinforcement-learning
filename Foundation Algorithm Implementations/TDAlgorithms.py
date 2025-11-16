from MDP import Policy
import gymnasium as gym
import random

class TDAlgorithms():
    def __init__(self):
        pass

    # Goal: learn value function online from experience under policy \pi
    def td_learning(self, policy: Policy, max_iter = 10, gamma = 0.7, alpha = 0.01):
        value_function = [0 for _ in range(64)]
        for iteration in range(max_iter):
            env = gym.make(
                "FrozenLake-v1",
                map_name="8x8",
                is_slippery=False,
                render_mode="human")
            state = env.reset()[0]
            terminated = False
            truncated = False
            while(not terminated and not truncated):
                actions, _ = policy.act(state)
                action = random.choice(actions)
                new_state, reward, terminated, truncated, _= env.step(action.value)
                value_function[state] = value_function[state] + alpha * (reward + gamma * value_function[new_state] - value_function[state])
                state = new_state
            env.close()

            print(f"Iteration {iteration} \n{value_function}")

    def sarsa(self):
        pass

    def Q_learning(self):
        pass
