# part1_2.py: Project 3 Part 1 script
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
- Complete the value_iteration method below
- Please write abundant comments and write neat code
- You can write any number of local functions
- More implementation details in the Function comments
"""

def value_iteration(env: mdp_env, agent: mdp_agent, eps: float, max_iter = 1000) -> np.ndarray:
	"""
	value_iteration method implements VALUE ITERATION MDP solver,
	shown in AIMA (3ed pg 653). The goal is to produce an optimal policy
	for any given mdp environment.

	Inputs
	---------------
		agent: The MDP solving agent (mdp_agent)
		env:   The MDP environment (mdp_env)
		eps:   Max error allowed in the utility of a state
		max_iter: Max iterations for the algorithm

	Outputs
	---------------
		policy: A list/array of actions for each state
				(Terminal states can have any random action)
	<agent>  Implicitly, you are populating the utlity matrix of
				the mdp agent. Do not return this function.
	"""
	policy = np.empty_like(env.states)
	agent.utility = np.zeros([len(env.states), 1])

	## START: Student code
	for _ in range(max_iter):
		delta = 0.0
		utility = np.copy(agent.utility)

		# Loop over all the states and update U
		for s in env.states:
			s_idx = int(s)

			action_values = []	
			reward = env.rewards[s_idx]
			
			# For all the actions in this state, compute utility
			for a in env.actions:
				probs = env.transition_model[s_idx, :, a]

				expected_utility = np.sum(probs * agent.utility[:, 0])
				action_values.append(expected_utility)

			# Select the max action and use it to comput the next value
			best_action_value = max(action_values)
			new_value = reward + agent.gamma * best_action_value

			# Update the utility with the new value and update delta
			utility[s, 0] = new_value
			delta = max(delta, abs(new_value - agent.utility[s, 0]))

		agent.utility = utility


		if delta < eps * (1 - agent.gamma) / agent.gamma:
			break

	# Extract policy from utility
	for s in env.states:
		s_idx = int(s)
		if s_idx in env.terminal:
			policy[s_idx] = 0  # arbitrary
			continue

		action_values = []
		# Compute expected utility for each action.
		for a in env.actions:
			q_value = np.sum(env.transition_model[s, :, a] * agent.utility[:, 0])
			action_values.append(q_value)
		# Choose the action with the highest expected utility.
		policy[s] = np.argmax(action_values)
	## END Student code
	print(policy)
	return policy
