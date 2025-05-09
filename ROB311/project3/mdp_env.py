# mdp_env.py: Project 3
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 3

import numpy as np

class mdp_env:
    """
        mdp_env class stores all the enviroment related functions and attributes. This describes
        an environment that follows the markov assumption

        Attributes
        ----------------
            states:     1D tuple of unique IDs (unsigned int) representing states
            actions:    1D tuple of unique IDs (unsigned int) repsenting available actions
            terminal:   1D tuple of states (UIDs) that are terminal
            rewards:    1D tuple of rewards for reaching each state
    """
    def __init__(self, states: tuple, actions: tuple, terminal: tuple, rewards: tuple):

        # Check if terminal states are valid
        for t in terminal:
            if t not in states:
                raise ValueError(f'mdp_env: init: Invalid terminal states: {terminal}')

        # Check if states and reward sizes are valid
        assert len(states) == len(rewards)

        self.states = states
        self.actions = actions
        self.terminal = terminal
        self.rewards = rewards