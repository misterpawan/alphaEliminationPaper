% load test_output_aelim/3.mat;
% load outputs_mcts_suite_sparse/ex2.mat;
% load outputs_mcts_suite_sparse/Erdos991.mat;
% load outputs_mcts_suite_sparse/hor_131.mat
% load outputs_mcts_suite_sparse/olm500.mat
% load outputs_mcts_suite_sparse/oscil_dcop_29.mat
% load outputs_mcts_suite_sparse/mbeause.mat
% load outputs_mcts_suite_sparse/tomography.mat
% load outputs_mcts_suite_sparse/mbeaflw.mat
load outputs_mcts_suite_sparse/bcsstm20.mat
nnz(matrix)

tic;
[L1, U1] = lu(matrix);
toc;
disp('Naive')
nnz(L1) + nnz(U1)

% [L2, U2] = lu_nopivot(P_mcts * matrix);
tic;
[L2, U2, P2, Q2] = lu(P_mcts * matrix, 0);
toc;
disp('MCTS')
nnz(L2) + nnz(U2)

tic;
P = colamd(matrix);
toc;
tic;
[L3, U3, P3, Q3] = lu(matrix(:, P));
toc;
disp('ColMD')
nnz(L3) + nnz(U3)

tic;
r = symrcm(matrix);
toc;
tic;
[L4, U4, P4, Q4] = lu(matrix(r, r));
toc;
disp('RCM')
nnz(L4) + nnz(U4)

tic;
m = symamd(matrix);
toc;
tic;
[L5, U5, P5, Q5] = lu(matrix(m, m));
toc;
% [L5, U5] = lu_nopivot(matrix(m, m));
disp('SymAMD')
nnz(L5) + nnz(U5)

% tic;
% m = dissect(matrix);
% toc;
% [L5, U5] = lu(matrix(m, m));
% disp('Nested Dissection')
% nnz(L5) + nnz(U5)
