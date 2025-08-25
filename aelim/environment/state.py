import numpy as np
import torch

from copy import deepcopy


class MatrixState:
    """
    Represents the State of the system when transforming a circuit. This holds the reference
    copy of the environment and the state of the transformation (even within a step).

    :param node_to_qubit: The mapping array, tau
    """

    def __init__(self, matrix, row=0):
        """
        Gets the state the DQN starts on. Randomly initializes the mapping if not specified
        otherwise, and sets the progress to 0 and gets the first gates to be scheduled.
        :return: list, [(n1, n2) next gates we can schedule]
        """
        self.matrix = matrix
        self.dimensions = matrix.shape
        self.elim_row = row
        self.done = False
        # The state must have access to the overall environment
        # self.circuit = circuit
        # self.device = device
        # assert len(circuit) == len(
        #     device), "All qubits on target device or not used, or too many are used"
        # The starting state should be setup right
        # self._node_to_qubit = self.device.allocate(self.circuit) \
        #     if node_to_qubit is None else node_to_qubit
        # self._qubit_targets = np.array([targets[0] if len(targets) > 0 else -1 for targets in self.circuit.circuit]) \
        #     if qubit_targets is None else qubit_targets
        # self._circuit_progress = np.zeros(len(self.circuit), dtype=np.int) \
        #     if circuit_progress is None else circuit_progress
        # self._locked_edges = np.zeros(len(self.device.edges), dtype=np.int) \
        #     if locked_edges is None else locked_edges

    def execute_elimination(self, index):
        # print(self.matrix, self.elim_row, index)
        self.matrix[[self.elim_row, index]
                    ] = self.matrix[[index, self.elim_row]]
        # prev_nonzeros = np.count_nonzero(self.matrix)
        if self.matrix[self.elim_row, self.elim_row] != 0:
            for i in range(self.elim_row+1, self.dimensions[0]):
                if self.matrix[i][self.elim_row] != 0:
                    self.matrix[i] -= self.matrix[self.elim_row] * \
                        self.matrix[i][self.elim_row] / \
                        self.matrix[self.elim_row][self.elim_row]
        self.matrix[np.abs(self.matrix) < 1e-10] = 0
        # final_nonzeros = np.count_nonzero(self.matrix)
        self.elim_row += 1

    def get_action_mask(self):
        a = np.array(
            [not (i < self.elim_row or (self.matrix[i, self.elim_row] == 0)) for i in range(self.dimensions[0])])
        if not any(a):
            a[self.elim_row] = 1
        return a

    def is_done(self):
        """
        Returns True iff each qubit has completed all of its interactions
        :return: bool, True if the entire circuit is executed
        """
        return self.elim_row >= (self.dimensions[0] - 1) or self.done

    # Other utility functions and properties

    def __copy__(self):
        """
        Makes a copy, keeping the reference to the same environment, but
        instantiating the rest of the state again.

        :return: State, a copy of the original, but independent of the first one, except env
        """
        return MatrixState(self.matrix, self.elim_row)

    # noinspection PyProtectedMember
    def __eq__(self, other):
        """
        Checks whether two state are identical

        :param other: State, the other state to compare against
        :return: True if they are the same, False otherwise
        """
        return (self.matrix == other.matrix).all() and self.elim_row == other.elim_row

    def get_input_matrix(self, encoding=0):
        if encoding == 0:
            mask = self.matrix != 0
            mat = deepcopy(self.matrix)
            mat[mask] = 1.0
            col_selector = torch.zeros((1, self.dimensions[1]))
            col_selector[0, self.elim_row] = 1.0
            return torch.vstack((torch.FloatTensor(mat), col_selector))
