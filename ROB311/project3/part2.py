# part2.py: Project 3 Part 2 script
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 3

import numpy as np
from mdp_env import mdp_env
from mdp_agent import mdp_agent


## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the policy_iteration method below
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""


def policy_iteration(env: mdp_env, agent: mdp_agent, max_iter = 1000) -> np.ndarray:
    """
    policy_iteration method implements VALUE ITERATION MDP solver,
    shown in AIMA (3ed pg 657). The goal is to produce an optimal policy
    for any given mdp environment.

    Inputs-
        agent: The MDP solving agent (mdp_agent)
        env:   The MDP environment (mdp_env)
        max_iter: Max iterations for the algorithm

    Outputs -
        policy: A list/array of actions for each state
                (Terminal states can have any random action)
       <agent>  Implicitly, you are populating the utlity matrix of
                the mdp agent. Do not return this function.
    """
    # np.random.seed(1) # TODO: Remove this

    policy = np.random.randint(len(env.actions), size=(len(env.states), ))
    agent.utility = np.zeros([len(env.states), 1])

    ## START: Student code
    num_states = len(env.states)
    num_actions = len(env.actions)
    
    # Outer loop: policy iteration. Stop when the policy is stable or max_iter is reached.
    for _ in range(max_iter):
        # ---- Policy Evaluation ----
        # Use a small threshold for evaluation convergence
        eval_threshold = 1e-3
        while True:
            delta = 0.0
            new_utility = np.copy(agent.utility)
            # Evaluate utility for each state under the current policy
            for s in range(num_states):
                # If the state is terminal, utility is simply its reward.
                if env.states[s] in env.terminal:
                    new_utility[s, 0] = env.rewards[s]
                else:
                    # Get the action specified by the current policy
                    a = policy[s]
                    # Compute the expected utility for state s following action a
                    expected_util = 0.0
                    for s_next in range(num_states):
                        # env.transition_model[s, s_next, a] is the probability of moving from s to s_next given a.
                        expected_util += env.transition_model[s, s_next, a] * agent.utility[s_next, 0]
                    new_utility[s, 0] = env.rewards[s] + agent.gamma * expected_util
                
                # Track the maximum change across states
                delta = max(delta, abs(new_utility[s, 0] - agent.utility[s, 0]))
            # Update the agent's utility with the newly computed utilities.
            agent.utility = new_utility
            # Break out of the evaluation loop if the change is below threshold.
            if delta < eval_threshold:
                break
        
        # ---- Policy Improvement ----
        policy_stable = True  # Assume policy is stable until we find a change
        for s in range(num_states):
            # For terminal states, policy improvement is not needed.
            if env.states[s] in env.terminal:
                continue
            
            current_action = policy[s]
            best_action = current_action
            best_action_value = -float("inf")
            # Evaluate each possible action for state s.
            for a in range(num_actions):
                q_value = 0.0
                for s_next in range(num_states):
                    q_value += env.transition_model[s, s_next, a] * agent.utility[s_next, 0]
                # If this action yields a higher expected utility, choose it.
                if q_value > best_action_value:
                    best_action_value = q_value
                    best_action = a
            
            # If the best action is different from the current action, update the policy.
            if best_action != current_action:
                policy[s] = best_action
                policy_stable = False
        
        # If no changes were made in policy improvement, the policy is optimal.
        if policy_stable:
            break
    ## END: Student code

    return policy