
from aima.search import Problem, exp_schedule, hill_climbing, simulated_annealing
import random

class NQueensProblemHC(Problem):
    """
    A Problem class to use with hill climbing, etc. Here, a state is a tuple
    where the c'th entry gives the row occupied by the queen in column c.
    
    We start with a random positioning of the queens, one per column.

    An action is represented as a (r, c) pair, indicating we move the queen in 
    column c to row r. Thus, from each state, there are 8x7 actions. 
    """

    def __init__(self, n):
        super().__init__(tuple(random.randrange(n) for _ in range(n)))
        self.N = n

    def actions(self, state):
        return [(r, c) for c in range(self.N) 
                       for r in range(self.N) if r != state[c]]
        

    def result(self, state, action):
        """Place the next queen at the given row."""
        new = list(state)
        new[action[1]] = action[0]
        return tuple(new)

    def conflicted(self, state, row, col):
        """Would placing a queen at (row, col) conflict with anything?"""
        return any(self.conflict(row, col, state[c], c)
                   for c in range(col))

    def conflict(self, row1, col1, row2, col2):
        """Would putting two queens in (row1, col1) and (row2, col2) conflict?"""
        return (row1 == row2 or  # same row
                col1 == col2 or  # same column
                row1 - col1 == row2 - col2 or  # same \ diagonal
                row1 + col1 == row2 + col2)  # same / diagonal

    def goal_test(self, state):
        """Check if all columns filled, no conflicts."""
        return not any(self.conflicted(state, state[col], col)
                       for col in range(len(state)))

    def value(self, state):
        """Return number of conflicting queens for a given node"""
        num_conflicts = 0
        for c1 in range(self.N):
            for c2 in range(c1+1, self.N):
                num_conflicts += self.conflict(state[c1], c1, state[c2], c2)

        return -num_conflicts

prob = NQueensProblemHC(8)
state = hill_climbing(prob)
print(state)
print(prob.value(state))

prob2 = NQueensProblemHC(8)
state = simulated_annealing(prob, schedule=exp_schedule(k=100, lam=0.02, limit=400))
print(state)
print(prob2.value(state))