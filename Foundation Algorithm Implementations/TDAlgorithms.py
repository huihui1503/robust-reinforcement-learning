from MDP import Policy, Action
import gymnasium as gym
import random
import numpy as np

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
                is_slippery=False)
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

    def explore_actions(self, greedy_action,num_actions, epsilon) -> Action:
        actions = []
        probs = []
        for i in range(num_actions):
            prob = epsilon / num_actions
            action = Action(i)
            if i == greedy_action:
                prob += 1 - epsilon
            actions.append(action)
            probs.append(prob)

        chosen_action = random.choices(actions, weights=probs, k=1)[0]
        return chosen_action

    def epsilon_greedy(self, state, action_value_matrix, epsilon, num_actions = 4) -> Action:
        greedy_action = np.argmax(action_value_matrix[state])
        action = self.explore_actions(greedy_action, num_actions, epsilon)
        return action

    def sarsa(self, policy: Policy, max_iter = 10, gamma = 0.9, alpha = 0.01, lamda = 0.7):
        action_value_matrix = np.zeros((64,4))
        for iteration in range(max_iter):
            env = gym.make(
                "FrozenLake-v1",
                map_name="8x8",
                is_slippery=False,)
                # render_mode="human")
            state = env.reset()[0]
            terminated = False
            truncated = False

            eligibility_trace = np.zeros((64,4))
            action = random.choice(policy.act(state)[0])
            while(not terminated and not truncated):
                next_state, reward, terminated, truncated, _= env.step(action.value)
                next_action = self.epsilon_greedy(state, action_value_matrix, 1/(iteration + 1) * 1.0, 4)

                delta = reward + gamma * action_value_matrix[next_state][next_action.value] - action_value_matrix[state][action.value]
                eligibility_trace[state][action.value] += 1

                for s in range(64):
                    for a in range(4):
                        action_value_matrix[s][a] = action_value_matrix[s][a] + alpha * delta * action_value_matrix[s][a]
                        eligibility_trace[s][a] = gamma * lamda * eligibility_trace[s][a]
                state = next_state
                action = next_action
            env.close()

            print(f"Iteration {iteration} \n{action_value_matrix}")

        return action_value_matrix

        

    def greedy_policy(self, state, action_value_matrix) -> Action:
        greedy_action = np.argmax(action_value_matrix[state])
        return greedy_action
    
    def Q_learning(self, policy: Policy, max_iter = 10, gamma = 0.9, alpha = 0.01, lamda = 0.7):
        action_value_matrix = np.zeros((64,4))
        for iteration in range(max_iter):
            env = gym.make(
                "FrozenLake-v1",
                map_name="8x8",
                is_slippery=False,)
                # render_mode="human")
            state = env.reset()[0]
            terminated = False
            truncated = False

            eligibility_trace = np.zeros((64,4))
            action = random.choice(policy.act(state)[0])
            while(not terminated and not truncated):
                next_state, reward, terminated, truncated, _= env.step(action.value)
                next_action = self.epsilon_greedy(state, action_value_matrix, 1/(iteration + 1) * 1.0, 4)
                greedy_action = self.greedy_policy(next_state, action_value_matrix)
                delta = reward + gamma * action_value_matrix[next_state][greedy_action.value] - action_value_matrix[state][action.value]
                eligibility_trace[state][action.value] += 1

                for s in range(64):
                    for a in range(4):
                        action_value_matrix[s][a] = action_value_matrix[s][a] + alpha * delta * action_value_matrix[s][a]
                        eligibility_trace[s][a] = gamma * lamda * eligibility_trace[s][a]
                state = next_state
                action = next_action
            env.close()

            print(f"Iteration {iteration} \n{action_value_matrix}")

        return action_value_matrix

        
