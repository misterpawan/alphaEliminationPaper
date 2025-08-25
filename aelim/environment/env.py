import copy
import collections

import numpy as np
import torch

from ..environment.state import MatrixState
from ..hyperparams import *


Moment = collections.namedtuple('Moment', ['cnots', 'swaps', 'reward'])


def step(action, input_state: MatrixState, eval=False):
    """
    Takes one step in the environment.
    This is actually the combination of 2 steps, the swaps in the current step and
    the cnot and setup for the next step.
    :param action: list of bool, whether we want to swap on each of the hardware connected nodes
    :param input_state: State, the state in the previous step
    :return: state, the state in the upcoming step
    :return: reward, the reward obtained from the operations in the current step
    :return: done, True if execution is complete, False otherwise
    :return: debugging output, Moment containing the gates executed and the reward obtained
    """

    # TODO: During Eval, no need to count
    state: MatrixState = copy.deepcopy(input_state)
    swap_pos = np.argmax(action)
    cur_elim_pos = state.elim_row
    # Log this
    if swap_pos < cur_elim_pos:
        state.done = True
        return state, REWARD_WRONG_MOVE, 1, "Selected row before elim"
    if swap_pos != cur_elim_pos and state.matrix[swap_pos, cur_elim_pos] == 0:
        state.done = True
        return state, REWARD_WRONG_MOVE, 1, "Selected row before elim"
    state.execute_elimination(swap_pos)
    nonzeros_created = np.count_nonzero(
        input_state.matrix) - np.count_nonzero(state.matrix)
    reward = nonzeros_created * REWARD_NON_ZEROS
    done = state.is_done()
    debugging_output = f"Performed Gaussian Elim after swapping row {cur_elim_pos} for row {swap_pos}"
    # print("Init-matrix", input_state.matrix)
    # print("Elim Row", cur_elim_pos)
    # print("Final-matrix", state.matrix)
    # print("Reward", reward)
    # a = 0
    # input(0)
    return state, reward, done, debugging_output


def evaluate(action, input_state: MatrixState):
    """
    Takes one step in the environment
    :param action: list of bool, whether we want to swap on each of the hardware connected nodes
    :param input_state: State, the state in the previous step
    :return: state, the state in the upcoming step
    :return: reward, the reward obtained from the operations in the current step
    :return: done, True if execution is complete, False otherwise
    :return: debugging output, Moment containing the gates executed and the reward obtained
    """
    # assert not np.any(np.bitwise_and(
    #     input_state.locked_edges, action)), "Bad Action"
    _next_state, reward, _done, _debug = step(action, input_state, eval=True)
    return reward


# def lu_simple(A):
#     """Performs an LU Decomposition of A (which must be square)
#     into PA = LU. The function returns P, L and U."""
#     n = A.shape[0]

#     # Create zero matrices for L and U
#     L = [[0.0] * n for i in range(n)]
#     U = [[0.0] * n for i in range(n)]

#     # Create the pivot matrix P and the multipled matrix PA
#     # P = pivot_matrix(A)
#     # PA = mult_matrix(P, A)

#     # Perform the LU Decomposition
#     for j in range(n):
#         # All diagonal entries of L are set to unity
#         L[j][j] = 1.0

#         # LaTeX: u_{ij} = a_{ij} - \sum_{k=1}^{i-1} u_{kj} l_{ik}
#         for i in range(j+1):
#             s1 = sum(U[k][j] * L[i][k] for k in range(i))
#             U[i][j] = A[i][j] - s1

#         # LaTeX: l_{ij} = \frac{1}{u_{jj}} (a_{ij} - \sum_{k=1}^{j-1} u_{kj} l_{ik} )
#         for i in range(j, n):
#             s2 = sum(U[k][j] * L[i][k] for k in range(j))
#             L[i][j] = (A[i][j] - s2) / U[j][j]

#     return (L, U)


# # LU decomposition of square systems


# def gaussian_elimination(A):
#     n = A.shape[0]
#     for k in range(0, n-1):
#         if A[k, k] == 0:
#             return
#         # for i in range()
#         for i in range(k+1, n):
#             A[i, k] = A[i, k] / A[k, k]
#         for j in range(k+1, n):
#             for i in range(k+1, n):
#                 A[i, j] -= A[i, k] * A[k, j]


# def naive_lu_factor(A):
#     """
#         No pivoting.

#         Overwrite A with:
#             U (upper triangular) and (unit Lower triangular) L
#         Returns LU (Even though A is also overwritten)
#     """
#     n = A.shape[0]
#     for k in range(n-1):
#         for i in range(k+1, n):
#             A[i, k] = A[i, k]/A[k, k]      # " L[i,k] = A[i,k]/A[k,k] "
#             for j in range(k+1, n):
#                 A[i, j] -= A[i, k]*A[k, j]  # " U[i,j] -= L[i,k]*A[k,j] "

#     return A  # (if you want)


# def lu_factor(A):
#     """
#         LU factorization with partial pivorting

#         Overwrite A with:
#             U (upper triangular) and (unit Lower triangular) L
#         Return [LU,piv]
#             Where piv is 1d numpy array with row swap indices
#     """
#     n = A.shape[0]
#     piv = np.arange(0, n)
#     for k in range(n-1):

#         # piv
#         max_row_index = np.argmax(abs(A[k:n, k])) + k
#         piv[[k, max_row_index]] = piv[[max_row_index, k]]
#         A[[k, max_row_index]] = A[[max_row_index, k]]

#         # LU
#         for i in range(k+1, n):
#             A[i, k] = A[i, k]/A[k, k]
#             for j in range(k+1, n):
#                 A[i, j] -= A[i, k]*A[k, j]

#     return [A, piv]


# def ufsub(L, b):
#     """ Unit row oriented forward substitution """
#     for i in range(L.shape[0]):
#         for j in range(i):
#             b[i] -= L[i, j]*b[j]
#     return b


# def bsub(U, y):
#     """ Row oriented backward substitution """
#     for i in range(U.shape[0]-1, -1, -1):
#         for j in range(i+1, U.shape[1]):
#             y[i] -= U[i, j]*y[j]
#         y[i] = y[i]/U[i, i]
#     return y

#     # No partial pivoting
# LU = naive_lu_factor(A)
# y = ufsub(LU, b)
# x = bsub(LU, y)

# # Partial pivoting
# LU, piv = lu_factor(A)
# b = b[piv]
# y = ufsub(LU, b)
# x = bsub(LU, y)
