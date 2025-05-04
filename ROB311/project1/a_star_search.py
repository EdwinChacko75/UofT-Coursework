import queue
import numpy as np
from search_problems import Node, GridSearchProblem, get_random_grid_problem


def reconstruct_path(came_from, prioirty, state):
    path = [state]
    while state in came_from.keys():
        state = came_from[state]
        path.insert(0, state)
    return path
def a_star_search(problem):
    """
    Uses the A* algorithm to solve an instance of GridSearchProblem. Use the methods of GridSearchProblem along with
    structures and functions from the allowed imports (see above) to implement A*.

    :param problem: an instance of GridSearchProblem to solve
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by your search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    num_nodes_expanded = 0
    max_frontier_size = 0

    closed_list = set([problem.init_state])

    g_score = {problem.init_state: 0}
    f_score = {problem.init_state: problem.heuristic(problem.init_state)}
    came_from = {}

    open_set = queue.PriorityQueue()
    open_set._put((0, problem.init_state))
    open_set_lookup = set([problem.init_state])

    while open_set:
        prioirty, state  = open_set._get()
        open_set_lookup.remove(state)

        if state in problem.goal_states:
            return reconstruct_path(came_from, prioirty, state), num_nodes_expanded, max_frontier_size
        
        closed_list.add(state)

        for action in problem.get_actions(state):
            node = problem.transition(state, action)
            if node in closed_list:
                continue
            node_g = g_score.get(node, float('inf'))
            cost = g_score[state] + problem.action_cost(state, action, node)

            if cost < node_g:

                came_from[node] = state
                g_score[node] = cost
                f_score[node] = g_score[node] + problem.heuristic(node)
                if node not in open_set_lookup:
                    open_set_lookup.add(node)
                    open_set._put((f_score[node], node))

    raise RuntimeError("No valid path found.")


def search_phase_transition():
    """
    Simply fill in the prob. of occupancy values for the 'phase transition' and peak nodes expanded within 0.05. You do
    NOT need to submit your code that determines the values here: that should be computed on your own machine. Simply
    fill in the values!

    :return: tuple containing (transition_start_probability, transition_end_probability, peak_probability)
    """
    ####
    #   REPLACE THESE VALUES
    ####
    transition_start_probability = 0.3
    transition_end_probability = 0.4
    peak_nodes_expanded_probability = 0.35
    return transition_start_probability, transition_end_probability, peak_nodes_expanded_probability


if __name__ == '__main__':
    # Test your code here!
    # Create a random instance of GridSearchProblem
    p_occ = 0.4
    M = 100
    N = 100
    problem = get_random_grid_problem(p_occ, M, N)
    # Solve it
    path, num_nodes_expanded, max_frontier_size = a_star_search(problem)
    # Check the result
    correct = problem.check_solution(path)
    print("Solution is correct: {:}".format(correct))
    # Plot the result
    problem.plot_solution(path)

    # Experiment and compare with BFS