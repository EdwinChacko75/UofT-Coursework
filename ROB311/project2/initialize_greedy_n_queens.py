import numpy as np
### WARNING: DO NOT CHANGE THE NAME OF THIS FILE, ITS FUNCTION SIGNATURE OR IMPORT STATEMENTS


def initialize_greedy_n_queens(N: int) -> list:
    """
    This function takes an integer N and produces an initial assignment that greedily (in terms of minimizing conflicts)
    assigns the row for each successive column. Note that if placing the i-th column's queen in multiple row positions j
    produces the same minimal number of conflicts, then you must break the tie RANDOMLY! This strongly affects the
    algorithm's performance!

    Example:
    Input N = 4 might produce greedy_init = np.array([0, 3, 1, 2]), which represents the following "chessboard":

     _ _ _ _
    |Q|_|_|_|
    |_|_|Q|_|
    |_|_|_|Q|
    |_|Q|_|_|

    which has one diagonal conflict between its two rightmost columns.

    You many only use numpy, which is imported as np, for this question. Access all functions needed via this name (np)
    as any additional import statements will be removed by the autograder.

    :param N: integer representing the size of the NxN chessboard
    :return: numpy array of shape (N,) containing an initial solution using greedy min-conflicts (this may contain
    conflicts). The i-th entry's value j represents the row  given as 0 <= j < N.
    """
    greedy_init = np.zeros(N, dtype=int)
    # First queen goes in a random spot
    greedy_init[0] = np.random.randint(0, N)

    ### YOUR CODE GOES HERE

    for col in range(1, N):
        rows = np.arange(N)  
        
        previous_queens = greedy_init[:col]  
        previous_cols = np.arange(col)  
        
        # Row conflicts (r_i = r_j)
        row_conflicts = np.isin(rows, previous_queens)

        # Diagonal (|row - prev_row| == |col - prev_col|)
        diagonal_conflicts = np.any(
            np.abs(previous_queens[:, None] - rows) == np.abs(previous_cols[:, None] - col), axis=0
        )

        conflicts = row_conflicts + diagonal_conflicts

        min_conflict_rows = rows[conflicts == conflicts.min()]

        greedy_init[col] = np.random.choice(min_conflict_rows)

    return greedy_init


if __name__ == '__main__':
    # You can test your code here
    N=10
    greedy = initialize_greedy_n_queens(N)

    board = np.zeros((N,N))
    board[greedy, np.arange(N)] = 1
    print(board)
