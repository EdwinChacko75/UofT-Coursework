# part1_1.py: Project 3 Part 1 script
#
# --
# Artificial Intelligence
# ROB 311
# Programming Project 3

import numpy as np
from mdp_cleaning_task import cleaning_env

## WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS

"""
INSTRUCTIONS
-----------------
  - Complete the method  get_transition_model which creates the
    transition probability matrix for the cleanign robot problem desribed in the
    project document.
  - Please write abundant comments and write neat code
  - You can write any number of local functions
  - More implementation details in the Function comments
"""

def get_transition_model(env: cleaning_env) -> np.ndarray:
    """
    get_transition_model method creates a table of size (SxSxA) that represents the
    probability of the agent going from s1 to s2 while taking action a
    e.g. P[s1,s2,a] = 0.5
    This is the method that will be used by the cleaning environment (described in the
    project document) for populating its transition probability table

    Inputs
    --------------
        env: The cleaning environment

    Outputs
    --------------
        P: Matrix of size (SxSxA) specifying all of the transition probabilities.
    """

    P = np.zeros([len(env.states), len(env.states), len(env.actions)])

    ## START: Student Code
    S = len(env.states)

    # Compute the probabilities for each action
    P_R = 0.15 * np.eye(S) # Shape: [S, S, 1]
    P_R += 0.05 * np.eye(S, k=-1)
    P_R += 0.8 * np.eye(S, k=1)

    # This flips it, which works
    P_L = P_R.T

    # Assign them in the table
    P[:, : , 0] = P_L
    P[:, : , 1] = P_R
    # Terninal states have 0 probability

    P[0, :, :] = 0 
    P[S-1, :, :] = 0 

    for t in env.terminal:
      P[t, :, :] = 0
    ## END: Student code
    print(P[:,:,0])
    print(P[:,:,1])
    return P

