import gymnasium as gym
from MDP import Policy, Action, MDP
from DPAlgorithms import DPAlgorithms
from TDAlgorithms import TDAlgorithms

if __name__ == "__main__":
    deterministic_env = gym.make(
        "FrozenLake-v1",
        map_name="8x8",
        is_slippery=False,
        render_mode="human",
        )
    
    env_unwrapped = deterministic_env.unwrapped
    environment = env_unwrapped.P 
    
    mdp = MDP(environment=environment)
    random_policy = Policy.random_policy(mdp.get_action_space(),mdp.get_state_space())

    dpAlgorithms = DPAlgorithms(mdp=mdp)

    ### Policy Evaluation
    reward_matrix = dpAlgorithms.policy_evaluation(policy=random_policy, max_iter= 100)
    ### Policy Iteration
    new_policy, reward_matrix =  dpAlgorithms.policy_iteration(random_policy, max_iter = 100)
    ### Value Iteration 
    new_policy, reward_matrix = dpAlgorithms.value_iteration(max_iter = 100)

    tdAlgorithms = TDAlgorithms()

    tdAlgorithms.td_learning(
        policy=new_policy,
        max_iter=10,
        gamma=0.7,
        alpha=0.01)



