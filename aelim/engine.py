from copy import copy, deepcopy
import os

import numpy as np
import tqdm
import torch
import wandb
from scipy.sparse.linalg import splu
from scipy.sparse import csc_matrix
import time as time_lib

from .metas import CombinerAgent
from .environment.env import step
from .environment.state import MatrixState


def train_step(matrix, save_path, agent=CombinerAgent,
               #    training_steps=100000,
               use_wandb=False, train_model=True):

    # os.makedirs("./test/test_results", exist_ok=True)
    state = MatrixState(matrix)
    # solution_start, solution_moments = np.array(state.node_to_qubit), []
    # progress_bar = tqdm.tqdm(total=len(list(circuit.cirq.all_operations())))

    # state, total_reward, done, debugging_output = step(
    #     np.full(len(state.device.edges), False), state)
    # progress_bar.update(len(debugging_output.cnots))
    # solution_moments.append(debugging_output)
    # progress_bar.set_description(episode_name)
    total_reward = 0

    for time in range(1, min(state.dimensions[0], state.dimensions[1])):
        # print(f"Time Step {time} of MCTS")

        action = agent.act(state)
        # print("Action at step -", action)
        next_state, reward, done, debugging_output = step(action, state)
        total_reward += reward
        state = next_state
        # print(state.matrix)

        if done:
            if train_model:
                loss_v, loss_p = agent.replay()
                if use_wandb:
                    wandb.log(
                        {'Value Loss': loss_v, 'Policy Loss': loss_p, 'Reward': total_reward})
                torch.save(agent.model.state_dict(),
                           save_path)
            if use_wandb:
                wandb.log({
                    'Steps Taken': time})
            return True

    print("Did not work!!")
    return False


def test_step_base(matrix, agent=CombinerAgent,
                   use_wandb=False):

    state_naive = MatrixState(deepcopy(matrix))
    state = MatrixState(matrix)
    num_actions = min(state.dimensions[0], state.dimensions[1])

    for time in range(1, num_actions):

        action = agent.act_test(state)
        next_state, reward, done, debugging_output = step(action, state)
        state = next_state

        naive_action = np.zeros(num_actions)
        if not np.any(state_naive.get_action_mask()):
            naive_action[time-1] = 1
        else:
            naive_action[np.where(state_naive.get_action_mask())[0][0]] = 1
        next_state, _, _, _ = step(naive_action, state_naive)
        state_naive = next_state

        if done:
            break

    print(np.count_nonzero(state_naive.matrix)
          - np.count_nonzero(state.matrix))

    return np.count_nonzero(state_naive.matrix) - np.count_nonzero(state.matrix)


def test_step_colmd(matrix, P, agent=CombinerAgent,
                    use_wandb=False):

    # os.makedirs("./test/test_results", exist_ok=True)
    state_naive = MatrixState(deepcopy(matrix))
    state = MatrixState(matrix)
    num_actions = min(state.dimensions[0], state.dimensions[1])

    P_mcts = np.eye(state.dimensions[0], state.dimensions[1])
    start_time = time_lib.time()

    for time in range(1, num_actions):

        action = agent.act(state)

        next_state, reward, done, debugging_output = step(action, state)
        state = next_state

        chosen_act = np.argmax(action)
        P_mcts[[time-1, chosen_act]] = P_mcts[[chosen_act, time-1]]

        if done:
            if use_wandb:
                wandb.log({
                    'Steps Taken': time})
            break

    end_time = time_lib.time()
    print(f"Time taken: {(end_time-start_time)/1000} s")

    slu1 = splu(csc_matrix(np.matmul(matrix, P)), permc_spec="NATURAL")
    # slu1 = splu(csc_matrix(matrix))

    slu2 = splu(csc_matrix(np.matmul(P_mcts, matrix)), permc_spec="NATURAL",
                diag_pivot_thresh=0, options={"SymmetricMode": True})

    slu3 = splu(csc_matrix(matrix), permc_spec="NATURAL")

    print("Special - ", csc_matrix.count_nonzero(slu1.L), csc_matrix.count_nonzero(slu1.U),
          csc_matrix.count_nonzero(slu1.L) + csc_matrix.count_nonzero(slu1.U))
    print("MCTS - ", csc_matrix.count_nonzero(slu2.L), csc_matrix.count_nonzero(slu2.U),
          csc_matrix.count_nonzero(slu2.L) + csc_matrix.count_nonzero(slu2.U))
    print("Naive - ", csc_matrix.count_nonzero(slu3.L), csc_matrix.count_nonzero(slu3.U),
          csc_matrix.count_nonzero(slu3.L) + csc_matrix.count_nonzero(slu3.U))

    if np.count_nonzero(state_naive.matrix) > np.count_nonzero(state.matrix):
        return True
    else:
        return False


def test_step_export(matrix, agent=CombinerAgent,
                     use_wandb=False):

    state = MatrixState(matrix)
    num_actions = min(state.dimensions[0], state.dimensions[1])

    P_mcts = np.eye(state.dimensions[0], state.dimensions[1])
    start_time = time_lib.time()

    for time in range(1, num_actions):

        action = agent.act(state)

        next_state, reward, done, debugging_output = step(action, state)
        state = next_state

        chosen_act = np.argmax(action)
        P_mcts[[time-1, chosen_act]] = P_mcts[[chosen_act, time-1]]

        if done:
            break

    end_time = time_lib.time()
    print(f"Time taken: {(end_time-start_time)/1000} s")

    return csc_matrix(P_mcts)
