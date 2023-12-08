import cvxpy as cp
import numpy as np

def solve_qp(Q, c, G, h, A, b):
    # Create a variable to optimize
    x = cp.Variable(len(c))

    # Define the objective function
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)

    # Define the constraints
    constraints = [G @ x <= h, A @ x == b]

    # Define the problem and solve it
    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value

def train_svm(Q, p, G, h, A, b):
    # Solve the QP problem
    solution = solve_qp(Q, p, G, h, A, b)
    
    # Extract the weights and bias from the solution
    w = solution[:-1]
    b = solution[-1]
    
    return w, b

def svm_qp_formulation(X, y, C):
    n_samples, n_features = X.shape

    # Extend Q for slack variables
    # Changed: The Q matrix should be square with size equal to n_features + n_samples
    Q = np.zeros((n_features + n_samples, n_features + n_samples))
    Q[:n_features, :n_features] = np.eye(n_features)

    # Changed: p should be a vector of length n_features + n_samples
    p = np.hstack([np.zeros(n_features), C * np.ones(n_samples)])

    # Construct G matrix
    # Changed: Ensure the G matrix has correct dimensions
    tmp1 = np.diag(y) @ X
    tmp2 = np.ones((n_samples, 1))
    G_top = np.hstack([-tmp1, -tmp2, -np.eye(n_samples)])
    G_bottom = np.hstack([np.zeros((n_samples, n_features + 1)), -np.eye(n_samples)])
    G = np.vstack([G_top, G_bottom])

    # Construct h vector
    h = np.hstack([-np.ones(n_samples), np.zeros(n_samples)])

    # Construct A matrix and b vector
    # Changed: Ensure A and b have correct dimensions
    A = np.zeros((1, n_features + n_samples))
    b = np.zeros(1)

    return Q, p, G, h, A, b

# Continue with the rest of your functions and testing script...



# # Example usage
# C = 1.0  # Regularization parameter
# X_train = ...  # Training data
# y_train = ...  # Labels (should be -1 or 1)

# Q, p, G, h, A, b = svm_qp_formulation(X_train, y_train, C)
# w, b = train_svm(Q, p, G, h, A, b)

def train_multiclass_svm(X, y, n_classes, C):
    """
    Trains a multiclass SVM using the One-vs-Rest approach.

    :param X: Training data, shape (n_samples, n_features)
    :param y: Class labels, shape (n_samples,)
    :param n_classes: Number of classes
    :param C: Regularization parameter
    :return: A list of tuples, each containing the weights and bias of a binary SVM
    """
    svms = []
    for i in range(n_classes):
        # Create binary labels for the current class
        y_binary = np.where(y == i, 1, -1)
        Q, p, G, h, A, b = svm_qp_formulation(X, y_binary, C)
        w, b = train_svm(Q, p, G, h, A, b)
        svms.append((w, b))
    return svms

def svm_predict(svms, X):
    """
    Predicts class labels for samples in X.

    :param svms: A list of trained binary SVMs (weights and bias)
    :param X: Data to predict, shape (n_samples, n_features)
    :return: Predicted class labels, shape (n_samples,)
    """
    n_samples = X.shape[0]
    n_classes = len(svms)
    scores = np.zeros((n_samples, n_classes))

    for i, (w, b) in enumerate(svms):
        scores[:, i] = X.dot(w) + b

    return np.argmax(scores, axis=1)

# # Example usage
# # Assuming X_train and y_train are defined
# n_classes = np.unique(y_train).size
# C = 1.0  # Regularization parameter

# # Train the multiclass SVM
# svms = train_multiclass_svm(X_train, y_train, n_classes, C)

# # Predict on some test data X_test
# y_pred = svm_predict(svms, X_test)