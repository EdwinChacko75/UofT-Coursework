import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def min_conflicts_n_queens(initialization: list) -> (list, int):
    """
    Solve the N-queens problem with no conflicts (i.e. each row, column, and diagonal contains at most 1 queen).
    Given an initialization for the N-queens problem, which may contain conflicts, this function uses the min-conflicts
    heuristic(see AIMA, pg. 221) to produce a conflict-free solution.

    Be sure to break 'ties' (in terms of minimial conflicts produced by a placement in a row) randomly.
    You should have a hard limit of 1000 steps, as your algorithm should be able to find a solution in far fewer (this
    is assuming you implemented initialize_greedy_n_queens.py correctly).

    Return the solution and the number of steps taken as a tuple. You will only be graded on the solution, but the
    number of steps is useful for your debugging and learning. If this algorithm and your initialization algorithm are
    implemented correctly, you should only take an average of 50 steps for values of N up to 1e6.

    As usual, do not change the import statements at the top of the file. You may import your initialize_greedy_n_queens
    function for testing on your machine, but it will be removed on the autograder (our test script will import both of
    your functions).

    On failure to find a solution after 1000 steps, return the tuple ([], -1).

    :param initialization: numpy array of shape (N,) where the i-th entry is the row of the queen in the ith column (may
                           contain conflicts)

    :return: solution - numpy array of shape (N,) containing a-conflict free assignment of queens (i-th entry represents
    the row of the i-th column, indexed from 0 to N-1)
             num_steps - number of steps (i.e. reassignment of 1 queen's position) required to find the solution.
    """

    N = len(initialization)
    solution = initialization.copy()
    num_steps = 0
    max_steps = 1000

    for idx in range(max_steps):
        ## YOUR CODE HERE
        num_steps +=1

        # Pick a random row conflict, if it exists
        unique_elements, counts = np.unique(solution, return_counts=True)
        duplicates = unique_elements[counts > 1]
        row_conflicts = np.where(np.isin(solution, duplicates))[0]
        row_conflict_idx = np.random.choice(row_conflicts) if row_conflicts.size > 0 else -1

        diag_conflicts = np.abs(np.diff(solution)) == 1
        diag_conflicts = np.where(diag_conflicts)[0]
        diag_conflict_idx = np.random.choice(diag_conflicts) if diag_conflicts.size > 0 else -1
        # print(diag_conflict_idx)

        do_diag = np.random.randint(2)

        # print(diag_conflict_idx, row_conflict_idx)
        # randomly handle a conflict
        if diag_conflict_idx > -1 or row_conflict_idx > -1:
            if diag_conflict_idx > -1 and (do_diag or row_conflict_idx == -1):
                solution = handle_diag_conflict(solution, diag_conflict_idx)
            else:
                solution = handle_row_conflict(solution, row_conflict_idx)
        else:
            print("No conflicts found!", num_steps)
            return solution, num_steps


    # return solution, num_steps
    return [], -1

def handle_diag_conflict(solution, idx):
    N = len(solution)
    all_squares = np.arange(N)
    col = idx
    row = solution[idx]

    conflict_row = solution[idx + 1]
    conflict_region = np.arange(conflict_row-1, conflict_row + 2)
    valid_sqaures = np.setdiff1d(all_squares, conflict_region)

    min_conflicts = count_conflicts(solution, N)
    ties_conflicts = []
    for sq in valid_sqaures:
        solution[idx] = sq
        new_conflicts = count_conflicts(solution, N)

        if new_conflicts < min_conflicts:
            min_conflicts = new_conflicts
            ties_conflicts = [sq]
        elif new_conflicts == min_conflicts:
            ties_conflicts.append(sq)

    solution[col] = np.random.choice(ties_conflicts) if ties_conflicts else row
    return solution

def handle_row_conflict(solution, idx):
    # return solution
    N = len(solution)
    all_squares = np.arange(N)
    col = idx
    row = solution[idx]

    unique_elements = np.unique(solution)
    valid_sqaures = np.setdiff1d(all_squares, unique_elements)

    min_conflicts = count_conflicts(solution, N)
    ties_conflicts = []
    for sq in valid_sqaures:
        solution[idx] = sq
        new_conflicts = count_conflicts(solution, N)

        if new_conflicts < min_conflicts:
            min_conflicts = new_conflicts
            ties_conflicts = [sq]
        elif new_conflicts == min_conflicts:
            ties_conflicts.append(sq)

    solution[col] = np.random.choice(ties_conflicts) if ties_conflicts else row
    return solution



def count_conflicts(board, N):
    cols = np.arange(N)  
    
    previous_queens = board[:, None]
    previous_cols = cols[:, None]  
    
    row_conflicts = np.sum(previous_queens == board, axis=0) - 1  
    
    diag_conflicts = np.sum(
        np.abs(previous_queens - previous_queens.T) == np.abs(previous_cols - previous_cols.T),
        axis=0
    ) - 1  
    
    total_conflicts = row_conflicts + diag_conflicts
    
    return np.sum(total_conflicts)


if __name__ == '__main__':
    # Test your code here!
    from initialize_greedy_n_queens import initialize_greedy_n_queens
    from support import plot_n_queens_solution

    N = 8
    # Use this after implementing initialize_greedy_n_queens.py
    assignment_initial = initialize_greedy_n_queens(N)
    # Plot the initial greedy assignment
    # assignment_initial = np.array([0,1,2,3])

    plot_n_queens_solution(assignment_initial, "before.png")
    assignment_solved, n_steps = min_conflicts_n_queens(assignment_initial)
    # Plot the solution produced by your algorithm
    plot_n_queens_solution(assignment_solved, "after.png")
