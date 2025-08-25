import os
import logging
import argparse

import wandb
import torch
import numpy as np
from tqdm import tqdm
import scipy.io as sio
from scipy.sparse import csc_matrix

from .algorithms.deepmcts import MCTSAgent
from .models.graph_dual import GraphDualModel
from .memory.list import MemorySimple
from .engine import test_step_colmd, train_step, test_step_base, test_step_export
from .visualizers.greedy_schedulers import cirq_routing, qiskit_routing, tket_routing

logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', default="base",
                        help='Choose training and test dataset from small, large, full, random')
    parser.add_argument('--iterations', default=100, type=int,
                        help='Number of iterations to train for on generated matrices.')
    parser.add_argument('--train', action='store_const', default=False, const=True,
                        help='Whether the training loop should be run or just evaluation.')
    parser.add_argument('--wandb', action='store_const', default=False, const=True,
                        help='Whether to use WandB to log the results of experiments.')
    parser.add_argument('--search', default=200, type=int,
                        help='Number of iterations to search for before making a move.')
    args = parser.parse_args()

    n, m = 500, 500
    sparsity = 0.94

    train_iters = args.iterations
    test_iters = 100

    matrix_dims = [n, m]
    input_dims = [n + 1, m]

    model = GraphDualModel(input_dims, True)
    memory = MemorySimple(0)
    agent = MCTSAgent(model, memory, search_depth=args.search)

    # Other preferences
    if args.wandb:
        os.system("wandb login fce7fc7e8358a46498fb184aff6a4ac48b1a7454")
        wandb.init(project='alpha-elim',
                   name='colmd_comparison_n', save_code=False)
    save_path = f"results/{n}_{m}_{sparsity}_weights.h5"
    # print(os.path.exists(save_path))
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))

    # Run on diff matrices

    for i in tqdm(range(train_iters)):

        # print(f"Started Iteration {i}")
        matrix = np.random.random((n, m))
        matrix[matrix > sparsity] = 0
        # print(f"Matrix -")
        # print(matrix)

        train_step(matrix, save_path, agent,
                   use_wandb=args.wandb, train_model=args.train)

    if args.dataset == "base":
        count = 0
        for i in range(test_iters):
            # print(f"Started test Iteration {i}")
            matrix = np.random.random((n, m))
            matrix[matrix < sparsity] = 0
            # print(f"Matrix -")
            # print(matrix)
            # if matrix.shape[0] != n:
            #     fin_mat = np.eye(n)
            #     fin_mat[n-1-matrix.shape[0]:n, n-1-matrix.shape[0]:n] = matrix
            #     matrix = fin_mat

            diff = test_step_base(
                matrix, agent, use_wandb=args.wandb)
            if args.wandb:
                wandb.log({'Test Diff (higher better)': diff})
            print('Test Diff (higher better)', diff)
            # P_mat = np.eye(n, m)
            # test_step_special(matrix, P_mat, agent)
            if diff >= 0:
                count += 1
        print(f"Better in {count}/{test_iters}")
    elif args.dataset == "colmd":
        # test_folder = f'test/test_set_colmd_{n}_{n}_{sparsity}/'
        test_folder = f'test/test_set_colmd/'
        for file in os.scandir(test_folder):
            # print(file)
            data = sio.loadmat(file)
            matrix = data['S'].toarray()
            P_mat_arr = data['P']

            P_mat = []
            for num in P_mat_arr[0]:
                arr = [0 for i in range(matrix.shape[0])]
                arr[int(num)-1] = 1
                P_mat.append(arr)
            P_mat = np.array(P_mat).T

            if matrix.shape[0] != n:
                fin_mat = np.eye(n)
                fin_mat[n-matrix.shape[0]:n, n-matrix.shape[0]:n] = matrix
                matrix = fin_mat
                P_mat_temp = np.eye(n)
                P_mat_temp[n-P_mat.shape[0]:n, n-P_mat.shape[0]:n] = P_mat
                P_mat = P_mat_temp
                # print(P_mat.shape)

            # print(P_mat, matrix)

            # print(matrix, P_mat)
            print("File ", file)
            test_step_colmd(matrix, P_mat, agent)
            print("=========================")

    elif args.dataset == "export":

        for i in range(test_iters):
            export_path = f"../test_output_aelim/{i}.mat"
            matrix = np.random.random((n, m))
            matrix[matrix < sparsity] = 0

            i_lower = np.tril_indices(n, -1)
            matrix[i_lower] = matrix.T[i_lower]

            P = test_step_export(
                matrix, agent)
            print(np.count_nonzero(matrix), "()()")
            sio.savemat(
                export_path, {'matrix': csc_matrix(matrix), 'P_mcts': P})

    elif args.dataset == "mmexport":

        test_folder = f'./suite_sparse/'
        for file in os.scandir(test_folder):
            export_path = f"../test_output_aelim/{file.name}.mat"
            # print(file)
            try:
                data = sio.loadmat(file)
            except:
                print(file.name)
                continue
            # print(data['Problem'][0][0])
            # for element in data['Problem'][0][0]:
            #     if element.shape[0] > 2:
            #         matrix = element
            if data['Problem'][0][0][1].dtype == np.float64:
                matrix = data['Problem'][0][0][1]
            else:
                matrix = data['Problem'][0][0][2]
            # print(matrix)
            matrix = matrix.toarray()
            if matrix.shape[0] != n:
                fin_mat = np.eye(n)
                fin_mat[n-matrix.shape[0]:n, n-matrix.shape[0]:n] = matrix
                matrix = fin_mat

            # i_lower = np.tril_indices(n, -1)
            # matrix[i_lower] = matrix.T[i_lower]

            P = test_step_export(matrix, agent)
            print(np.count_nonzero(matrix), "()()")
            sio.savemat(
                export_path, {'matrix': csc_matrix(matrix), 'P_mcts': P})
