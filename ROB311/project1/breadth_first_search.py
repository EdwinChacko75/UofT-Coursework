from collections import deque
import numpy as np
from search_problems import Node, GraphSearchProblem

def breadth_first_search(problem):
    """
    Implement a simple breadth-first search algorithm that takes instances of SimpleSearchProblem (or its derived
    classes) and provides a valid and optimal path from the initial state to the goal state. Useful for testing your
    bidirectional and A* search algorithms.

    :param problem: instance of SimpleSearchProblem
    :return: path: a list of states (ints) describing the path from problem.init_state to problem.goal_state[0]
             num_nodes_expanded: number of nodes expanded by the search
             max_frontier_size: maximum frontier size during search
    """
    ####
    #   COMPLETE THIS CODE
    ####
    max_frontier_size = 0
    num_nodes_expanded = 0
    path = []

    visited = set([problem.init_state])

    source = Node(None, problem.init_state, None, 0)
    open_set = deque([source])


    goal = None
    while open_set:
        max_frontier_size = max(max_frontier_size, len(open_set))

        cur_node = open_set.popleft()

        if cur_node.state in problem.goal_states:
            goal = cur_node
            break

        num_nodes_expanded += 1
        
        for action in problem.get_actions(cur_node.state):
            node = problem.get_child_node(cur_node, action)
            if node.state not in visited:
                visited.add(node.state)
                open_set.append(node)

    if goal is None:
        return path, num_nodes_expanded, max_frontier_size
    
    while goal.parent is not None:
        path.append(goal.action[1])
        goal = goal.parent

    path.append(problem.init_state)
    path.reverse()

    # print(f"Number of nodes expanded: {num_nodes_expanded}")
    # print(f"Max frontier size: {max_frontier_size}")

    return path, num_nodes_expanded, max_frontier_size


if __name__ == '__main__':
    # Simple example
    goal_states = [0]
    init_state = 9
    V = np.arange(0, 10)
    E = np.array([[0, 1],
                  [1, 2],
                  [2, 3],
                  [3, 4],
                  [4, 5],
                  [5, 6],
                  [6, 7],
                  [7, 8],
                  [8, 9],
                  [0, 6],
                  [1, 7],
                  [2, 5],
                  [9, 4]])
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)

    # Use stanford_large_network_facebook_combined.txt to make your own test instances
    E = np.loadtxt('./stanford_large_network_facebook_combined.txt', dtype=int)
    V = np.unique(E)
    goal_states = [349]
    init_state = 0
    problem = GraphSearchProblem(goal_states, init_state, V, E)
    path, num_nodes_expanded, max_frontier_size = breadth_first_search(problem)
    correct = problem.check_graph_solution(path)
    print("Solution is correct: {:}".format(correct))
    print(path)