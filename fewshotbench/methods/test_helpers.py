# test_helpers.py
import unittest
import numpy as np
from helpers import solve_qp  # Importing the solve_qp function

class TestQuadraticSolver(unittest.TestCase):
    # Additional tests in test_helpers.py

    def test_infeasible_problem(self):
        # Define an infeasible problem
        Q = np.array([[2, 0], [0, 2]])
        c = np.array([-1, -1])
        G = np.array([[1, 0], [0, 1]])
        h = np.array([-1, -1])
        A = np.array([[1, 1]])
        b = np.array([2])
        x_opt = solve_qp(Q, c, G, h, A, b)
        self.assertIsNone(x_opt)  # Expecting None or a similar indicator for infeasibility

    def test_unbounded_problem(self):
        # Define an unbounded problem
        Q = np.zeros((2, 2))
        c = np.array([-1, -1])
        G = np.array([[0, 0], [0, 0]])  # No effective inequality constraints
        h = np.array([0, 0])
        A = np.array([[0, 0]])          # No effective equality constraints
        b = np.array([0])
        x_opt = solve_qp(Q, c, G, h, A, b)
        self.assertIsNone(x_opt)  # Expecting None or a similar indicator for unboundedness


    def test_large_problem(self):
        # Define a larger problem
        n = 10
        Q = np.eye(n)
        c = -np.ones(n)
        G = -np.eye(n)
        h = np.zeros(n)
        A = np.ones((1, n))
        b = np.array([n / 2])
        x_opt = solve_qp(Q, c, G, h, A, b)
        self.assertIsNotNone(x_opt)  # Expecting a valid solution

    def test_zero_matrices(self):
        # Define a problem with zero matrices
        Q = np.zeros((2, 2))
        c = np.zeros(2)
        G = -np.eye(2)
        h = np.zeros(2)
        A = np.array([[1, 1]])
        b = np.array([1])
        x_opt = solve_qp(Q, c, G, h, A, b)
        self.assertIsNotNone(x_opt)  # Expecting a valid solution

# [Rest of the test_helpers.py file]


if __name__ == '__main__':
    unittest.main()
