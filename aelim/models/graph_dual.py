import typing
import numpy as np

import torch

from ..environment.state import MatrixState


class NormActivation(torch.nn.Module):

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, tensor):
        tensor = tensor ** 2
        length = tensor.sum(dim=self.dim, keepdim=True)
        return tensor / length


class GraphDualModel(torch.nn.Module):

    # def __init__(self, dimensions, stop_move: bool = False):
    #     """
    #     Create the decision model for the given device topology
    #     :param device: the device object on which the agent should propose actions
    #     """
    #     super(GraphDualModel, self).__init__()
    #     self.dimensions = dimensions
    #     output_dims = [((dimensions[0])//2)//2,
    #                    ((dimensions[1])//2)//2]
    #     self.mlp = torch.nn.Sequential(
    #         # 1 * n * m -> 10 * n * m
    #         torch.nn.Conv2d(in_channels=1, out_channels=10,
    #                         kernel_size=3, padding=1),
    #         torch.nn.ReLU(),

    #         # 10 * n * m -> 10 * floor(n/2) * ceil(m/2)
    #         torch.nn.MaxPool2d(kernel_size=2),

    #         # 10 * floor(n/2) * ceil(m/2) -> 20 * ceil(n/2) * ceil(m/2)
    #         torch.nn.Conv2d(in_channels=10, out_channels=20,
    #                         kernel_size=5, padding=2),
    #         torch.nn.ReLU(),

    #         # 20 * floor(n/2) * ceil(m/2) -> 20 * ceil(ceil(n/2)/2) * ceil(ceil(m/2)/2)
    #         torch.nn.MaxPool2d(kernel_size=2),

    #         torch.nn.Flatten(),
    #         # 20 * floor(ceil(n/2)/2) * ceil(ceil(m/2)/2)
    #         torch.nn.Linear(20 * output_dims[0]
    #                         * output_dims[1], dimensions[0]*2),
    #         torch.nn.ReLU()
    #     )
    #     self.value_head = torch.nn.Sequential(
    #         torch.nn.Linear(dimensions[0]*2, 16),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(16, 1),
    #     )
    #     self.policy_head = torch.nn.Sequential(
    #         torch.nn.Linear(dimensions[0]*2, dimensions[1]),
    #         NormActivation(dim=-1),
    #     )
    #     self.optimizer = torch.optim.Adam(self.parameters())

    def __init__(self, dimensions, stop_move: bool = False):
        """
        Create the decision model for the given device topology
        :param device: the device object on which the agent should propose actions
        """
        super(GraphDualModel, self).__init__()
        self.dimensions = dimensions
        output_dims = [((dimensions[0])//2)//2,
                       ((dimensions[1])//2)//2]
        self.mlp = torch.nn.Sequential(
            # 1 * n * m -> 3 * n * m
            torch.nn.Conv2d(in_channels=1, out_channels=3,
                            kernel_size=3, padding=1),
            torch.nn.ReLU(),

            # 3 * n * m -> 6 * floor(n/2) * ceil(m/2)
            torch.nn.MaxPool2d(kernel_size=2),

            # 6 * floor(n/2) * ceil(m/2) -> 6 * ceil(n/2) * ceil(m/2)
            torch.nn.Conv2d(in_channels=3, out_channels=6,
                            kernel_size=5, padding=2),
            torch.nn.ReLU(),

            # 6 * floor(n/2) * ceil(m/2) -> 6 * ceil(ceil(n/2)/2) * ceil(ceil(m/2)/2)
            torch.nn.MaxPool2d(kernel_size=2),

            torch.nn.Flatten(),
            # 6 * floor(ceil(n/2)/2) * ceil(ceil(m/2)/2)
            torch.nn.Linear(6 * output_dims[0]
                            * output_dims[1], (dimensions[0]*2)//2),
            torch.nn.ReLU()
        )
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear((dimensions[0]*2)//2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        self.policy_head = torch.nn.Sequential(
            torch.nn.Linear((dimensions[0]*2)//2, dimensions[1]),
            NormActivation(dim=-1),
        )
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, state: MatrixState) -> typing.Tuple[int, np.ndarray]:
        """
        The callable for the model, does the forward propagation step
        :param state: input state of the circuit
        :return: the probability of each of the actions and value function for state
        """
        # x, remaining, locks = self.get_representation(state)
        # x = self.edge_conv(x, self.edges)
        # x = x.view(-1)
        # value_input = torch.cat([x, remaining, locks])
        # policy_input = torch.cat([x, locks])
        # policy = self.policy_head(policy_input)
        # value: int = self.value_head(value_input)
        # policy[-1] = -1e10  FIXME: Force this constraint for all other functions

        x = state.get_input_matrix()
        # print(x.shape)
        x = torch.unsqueeze(x, dim=0)
        # print(x.shape)
        # print(self.dimensions)
        x = torch.unsqueeze(x, dim=0)
        x = self.mlp(x)
        value = self.value_head(x)
        policy = self.policy_head(x)
        return value, policy

    # def get_representation(self, state: MatrixState):
    #     """
    #     Obtains the state representation
    #     :param state: the state of the circuit right now
    #     """
    #     nodes_to_target_nodes = state.target_nodes
    #     interaction_map = torch.zeros((len(self.device), len(self.device)))
    #     for idx, target in enumerate(nodes_to_target_nodes):
    #         if target == -1:
    #             continue
    #         interaction_map[idx, target] = 1

    #     remaining_targets = torch.from_numpy(state.remaining_targets)
    #     mutex_locks = torch.from_numpy(state.locked_edges)
    #     return interaction_map, remaining_targets, mutex_locks

    @staticmethod
    def _loss_p(predicted, target):
        loss = torch.sum(-target * ((1e-8 + predicted).log()))
        return loss

    @staticmethod
    def _loss_v(predicted, target):
        criterion = torch.nn.MSELoss()
        loss = criterion(predicted, target)
        return loss

    def fit(self, state, v, p):
        self.optimizer.zero_grad()
        self.train()
        v = v.reshape(1)
        pred_v, pred_p = self(state)
        # print(v, p)
        # print(pred_v[0], pred_p[0])
        v_loss = self._loss_v(pred_v[0], v)
        p_loss = self._loss_p(pred_p[0], p)
        loss = v_loss + p_loss
        loss.backward()
        self.optimizer.step()
        return v_loss.item(), p_loss.item()
