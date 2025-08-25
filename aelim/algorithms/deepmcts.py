"""
Monte Carlo Tree Search for asymmetric trees
CREDITS : Thomas Moerland, Delft University of Technology
"""

import copy
import typing as ty
import collections

import numpy as np
import torch
from copy import deepcopy

from ..metas import CombinerAgent
from ..environment.state import MatrixState
from ..environment.env import step, evaluate

MemoryItem = collections.namedtuple(
    'MemoryItem', ['state', 'reward', 'action', 'next_state', 'done'])


class MCTSAgent(CombinerAgent):

    class MCTSState:
        """
        State object representing the solution (boolean vector of swaps) as a MCTS node
        """

        HYPERPARAM_NOISE_ALPHA = 0.2
        HYPERPARAM_PRIOR_FRACTION = 0.25

        def __init__(self, state, model, r_previous=0, parent_state=None, parent_action=None):
            """
            Initialize a new state
            """
            self.state: MatrixState = state
            self.model = model
            self.parent_state, self.parent_action = parent_state, parent_action
            self.r_previous = r_previous
            self.num_actions = self.state.dimensions[0]
            # self.solution:

            self.rollout_reward = self.rollout() if self.parent_action is not None else 0.0
            # TODO : check this
            self.action_mask = self.state.get_action_mask()
            # print("Action Mask", self.state.matrix,
            #       self.state. elim_row, self.action_mask)

            self.n_value = torch.zeros(self.num_actions)
            self.q_value = torch.zeros(self.num_actions)
            self.w_value = torch.zeros(self.num_actions)
            self.child_states: ty.List[ty.Optional[MCTSAgent.MCTSState]] = [
                None for _ in range(self.num_actions + 1)]

            model.eval()
            with torch.no_grad():
                _value, self.priors = self.model(self.state)
                self.priors = self.priors.detach().numpy()
                self.priors += np.bitwise_not(self.action_mask) * -1e8
                self.priors = torch.flatten(torch.tensor(self.priors))
            noise = np.random.dirichlet(
                [self.HYPERPARAM_NOISE_ALPHA for _ in self.priors]) * self.action_mask
            self.priors = self.HYPERPARAM_PRIOR_FRACTION * \
                self.priors + (1 - self.HYPERPARAM_PRIOR_FRACTION) * noise

        def update_q(self, reward, index):
            """
            Updates the q-value for the state
            :param reward: The obtained total reward from this state
            :param index: the index of the action chosen for which the reward was provided

            n_value is the number of times a node visited
            q_value is the q function

            n += 1, w += reward, q = w / n -> this is being implicitly computed using the weighted average
            """
            self.w_value[index] += reward
            self.n_value[index] += 1
            self.q_value[index] = self.w_value[index]/self.n_value[index]

        def select(self, c=200) -> int:
            """
            Select one of the child actions based on UCT rule
            """
            n_visits = torch.sum(self.n_value).item()
            uct = self.q_value + \
                (self.priors * c * np.sqrt(n_visits + 0.001) / (self.n_value + 0.001))
            best_val = torch.max(uct)
            best_move_indices: torch.Tensor = torch.where(
                torch.eq(best_val, uct))[0]
            winner: int = np.random.choice(best_move_indices.numpy())
            return winner

        def choose(self) -> int:
            """
            Select one of the child actions based on the best q-value which is allowed
            """
            q_real = self.q_value + np.bitwise_not(self.action_mask) * -1e8
            best_val = torch.max(q_real)
            best_move_indices: torch.Tensor = torch.where(
                torch.eq(best_val, q_real))[0]
            winner: int = np.random.choice(best_move_indices.numpy())
            return winner

        def choose_test(self) -> int:
            """
            Select one of the child actions based on the best q-value which is allowed
            """
            # q_real =  + np.bitwise_not(self.action_mask) * -1e8
            best_val = torch.max(self.priors)
            best_move_indices: torch.Tensor = torch.where(
                torch.eq(best_val, self.priors))[0]
            winner: int = np.random.choice(best_move_indices.numpy())
            return winner

        def rollout(self, num_rollouts=None):  # TODO: Benchmark this on 100 rollouts
            """
            performs R random rollout, the total reward in each rollout is computed.
            returns: mean across the R random rollouts.
            """
            if num_rollouts is None:
                with torch.no_grad():
                    self.model.eval()
                    self.rollout_reward, _priors = self.model(self.state)
                return self.rollout_reward.item()
            else:
                pass

    """
    Monte Carlo Tree Search combiner object for evaluating the combination of moves
    that will form one step of the simulation.
    This at the moment does not look into the future steps, just calls an evaluator
    """

    HYPERPARAM_DISCOUNT_FACTOR = 0.95
    HYPERPARAM_EXPLORE_C = 100
    HYPERPARAM_POLICY_TEMPERATURE = 0

    def __init__(self, model, memory, search_depth=100):
        super().__init__(model)
        self.model = model
        self.root: ty.Optional[MCTSAgent.MCTSState] = None
        self.memory = memory
        self.search_depth = search_depth

    def search(self):
        """Perform the MCTS search from the root"""
        max_depth, mean_depth = 0, 0
        n_mcts = 0
        # print(self.root.state.matrix)
        for _ in range(50):
            # print(f"Serach Iteration {_}")
            n_mcts += 1
            mcts_state: MCTSAgent.MCTSState = self.root  # reset to root for new trace
            # input(str(self.root.n_value) + " " +
            #       str(self.root.q_value))  # To Debug the tree
            depth = 0

            while True:
                depth += 1
                action_index: int = mcts_state.select(c=100)

                if mcts_state.child_states[action_index] is not None:
                    # MCTS Algorithm: SELECT STAGE
                    mcts_state = mcts_state.child_states[action_index]
                    continue
                elif mcts_state.state.is_done():
                    break
                else:
                    # MCTS Algorithm: EXPAND STAGE
                    action = np.zeros(mcts_state.num_actions)
                    action[action_index] = 1
                    next_state, reward, _done, _debug = step(
                        action, mcts_state.state)
                    mcts_state.child_states[action_index] = MCTSAgent.MCTSState(
                        next_state, self.model,
                        r_previous=reward, parent_state=mcts_state, parent_action=action_index)

                    mcts_state = mcts_state.child_states[action_index]
                    break

            # MCTS Algorithm: BACKUP STAGE
            total_reward = mcts_state.rollout_reward
            while mcts_state.parent_action is not None:
                total_reward = mcts_state.r_previous + \
                    self.HYPERPARAM_DISCOUNT_FACTOR * total_reward
                mcts_state.parent_state.update_q(
                    total_reward, mcts_state.parent_action)
                mcts_state = mcts_state.parent_state

            max_depth = max(max_depth, depth)
            mean_depth += depth

        mean_depth /= n_mcts
        # print(self.root.state.matrix)
        # print("Searching Complete")
        return max_depth, mean_depth

    @ staticmethod
    def _stable_normalizer(x, temp=1.5):
        x = (x / torch.max(x)) ** temp
        return torch.abs(x / torch.sum(x))

    def act(self, state):
        """Process the output at the root node"""
        if self.root is None or self.root.state != state:
            self.root = MCTSAgent.MCTSState(state, self.model)
        else:
            self.root.parent_state = None
            self.root.parent_action = None

        self.search()
        self.memory.store(state,
                          torch.sum(
                              (self.root.n_value / torch.sum(self.root.n_value)) * self.root.q_value),
                          self._stable_normalizer(self.root.n_value))
        pos = self.root.choose()
        action = np.zeros(self.root.num_actions)
        action[pos] = 1
        return action

    def act_test(self, state):
        """Process the output at the root node"""
        if self.root is None or self.root.state != state:
            self.root = MCTSAgent.MCTSState(state, self.model)
        else:
            self.root.parent_state = None
            self.root.parent_action = None

        pos = self.root.choose_test()
        action = np.zeros(self.root.num_actions)
        action[pos] = 1
        return action

    # TODO: prioritized exp replay

    def replay(self):
        self.model.train()
        value_losses = []
        policy_losses = []
        for state, v, p in self.memory:
            loss_v, loss_p = self.model.fit(state, v, p)
            value_losses.append(loss_v)
            policy_losses.append(loss_p)
        self.memory.clear()
        return np.mean(value_losses), np.mean(policy_losses)
