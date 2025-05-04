import numpy as np
from mdp_cleaning_task import cleaning_env
from mdp_agent import mdp_agent
from part1_2 import value_iteration
from part1_1 import get_transition_model

def test_value_iteration():
    # Create the environment
    env = cleaning_env(rewards=(10, 0, 0, 0, 0, 2))
    env.init_stochatic_model(get_transition_model)  # Initialize transition model

    # Create the agent
    agent = mdp_agent(gamma=0.9)  # Use a discount factor

    # Assign the generated transition model to match expected field in agent code
    env.transitions = env.transition_model  # Compatibility with value_iteration()

    # Print environment structure and transition matrix
    env.print_env()
    env.print_transition_model()

    # Run value iteration
    eps = 1e-3
    policy = value_iteration(env, agent, eps)

    # Print results
    print("------------- Results --------------")
    print("Optimal Policy (0 = LEFT, 1 = RIGHT):")
    for s in env.states:
        print(f"State {s}: Action {policy[s]} ({env.action_names[policy[s]]})")

    print("\nUtility values:")
    for s in env.states:
        print(f"U({s}) = {agent.utility[s][0]:.4f}")

if __name__ == "__main__":
    test_value_iteration()
