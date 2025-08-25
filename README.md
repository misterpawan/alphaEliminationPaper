# Alpha Elimination: Learning to Reduce Fill-In in Sparse LU

Reduce fill-in during sparse matrix factorization using a deep-RL approach that couples a neural policy with Monte-Carlo Tree Search (MCTS). The method learns a reordering policy that selects elimination rows/columns to minimize the number of nonzeros created during LU, improving both runtime and memory footprint compared to classic heuristics like AMD/RCM/COLAMD/METIS.

> Paper: **Alpha Elimination: Using Deep Reinforcement Learning to Reduce Fill-In during Sparse Matrix Decomposition** (Dasgupta & Kumar, 2023). arXiv:2310.09852.

---

## Table of Contents

- [Highlights](#highlights)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Reproducing Paper Results](#reproducing-paper-results)
- [Datasets](#datasets)
- [Configuration](#configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Baselines](#baselines)
- [Results (Expected)](#results-expected)
- [Cite](#cite)
- [License](#license)

---

## Highlights

- **Problem**: Fill-in minimization in sparse LU via row/column elimination ordering (NP-hard).  
- **Approach**: Single-agent RL with **MCTS + neural policy/value** to select the next elimination choice.  
- **Outcome**: Fewer nonzeros in \(L\) and \(U\) vs. popular heuristics, with comparable wall-clock.  

---

## Repository Structure

```
alphaEliminationPaper/
├─ src/
│  ├─ alpha_elimination/
│  │  ├─ envs/           # Sparse matrix elimination environment
│  │  ├─ mcts/           # MCTS search, node/edge, UCT, rollout
│  │  ├─ models/         # Policy/Value networks
│  │  ├─ utils/          # io, logging, metrics, seeding
│  │  └─ train.py        # Training loop (self-play or dataset-driven)
│  └─ cli/               # Command-line entry points
├─ configs/
│  ├─ default.yaml       # Training & model hyperparameters
│  ├─ eval.yaml          # Evaluation settings
│  └─ baselines.yaml     # Heuristic baselines configuration
├─ data/
│  ├─ suitesparse/       # (optional) matrices or list files
│  └─ toy/               # small matrices for smoke tests
├─ scripts/
│  ├─ prepare_data.py    # Download/convert matrices
│  ├─ run_train.sh       # Example training runner
│  └─ run_eval.sh        # Example evaluation runner
├─ notebooks/
│  └─ analysis.ipynb     # Plots, ablations
├─ requirements.txt
└─ README.md
```

---

## Installation

Tested on Python 3.9+.

```bash
git clone https://github.com/misterpawan/alphaEliminationPaper.git
cd alphaEliminationPaper

# (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

If you use GPU acceleration for the factorization backends or PyTorch, ensure the matching CUDA toolkit is installed.

---

## Quickstart

### 1) Smoke test on toy matrices

```bash
# Train a small policy/value model with MCTS on toy 20x20 matrices
python -m src.alpha_elimination.train \
  --config configs/default.yaml \
  --data_dir data/toy \
  --log_dir runs/toy
```

### 2) Evaluate vs. baselines

```bash
python -m src.alpha_elimination.eval \
  --config configs/eval.yaml \
  --checkpoint runs/toy/best.ckpt \
  --mat_list data/toy/list.txt \
  --baselines AMD RCM COLAMD METIS \
  --out results/toy_eval.json
```

### 3) Plot results

```bash
python -m src.alpha_elimination.analysis.plot \
  --results results/toy_eval.json \
  --savefig results/toy_eval.png
```

---

## Reproducing Paper Results

1. **Fetch matrices** (e.g., SuiteSparse selection used in the paper):

```bash
python scripts/prepare_data.py --suite suitesparse --out data/suitesparse
```

2. **Train**:

```bash
bash scripts/run_train.sh \
  --config configs/default.yaml \
  --data_dir data/suitesparse \
  --log_dir runs/suitesparse
```

3. **Evaluate** (on held-out matrices):

```bash
bash scripts/run_eval.sh \
  --config configs/eval.yaml \
  --checkpoint runs/suitesparse/best.ckpt \
  --mat_list data/suitesparse/test_list.txt \
  --out results/suitesparse_eval.json
```

> Tip: For determinism, set seeds in the config and export `OMP_NUM_THREADS=1`.

---

## Datasets

- **SuiteSparse Matrix Collection** (recommended): real-world sparse matrices across domains.  
  Use `scripts/prepare_data.py` or provide your own matrix list in Matrix Market (`.mtx`) or CSR/COO NumPy formats.
- **Toy matrices**: small synthetic instances in `data/toy/` for quick checks.

---

## Configuration

All hyperparameters live in YAML files under `configs/`:

- `default.yaml`: network size, optimizer, learning rate schedule, MCTS parameters (simulations, UCT constant), replay buffer, curriculum, etc.
- `eval.yaml`: evaluation batch size, factorization backend, metrics to compute.
- `baselines.yaml`: toggles/params for AMD/RCM/COLAMD/METIS backends where available.

Override any key from the CLI:

```bash
python -m src.alpha_elimination.train \
  --config configs/default.yaml \
  trainer.max_steps=200000 \
  mcts.simulations=1600 \
  model.hidden_dim=256
```

---

## Evaluation Metrics

For a matrix \(A\) and elimination order \(\pi\):

- **Fill-in (nnz)**: \( \text{nnz}(L)+\text{nnz}(U) \) (excluding diagonal if desired).
- **Fill-ratio**: \( \frac{\text{nnz}(L)+\text{nnz}(U)}{\text{nnz}(A)} \).
- **Runtime**: ordering time + factorization time.
- **Memory proxy**: peak nnz during factorization / bytes allocated (if backend exposes it).

---

## Baselines

We compare the learned ordering to common heuristics (as available in your sparse library):

- **AMD** (Approximate Minimum Degree)  
- **RCM** (Reverse Cuthill–McKee)  
- **COLAMD/CAMD**  
- **METIS** (graph partitioning based)

Enable/disable in `configs/baselines.yaml` or via CLI flags like `--baselines AMD RCM`. (Backend requirements vary by platform.)

---

## Results (Expected)

Empirically, the learned policy reduces fill-in (total nnz in \(L+U\)) versus standard heuristics, often with similar wall-clock cost when amortized. See the paper for quantitative tables and ablations on search depth, number of MCTS simulations, and network capacity.

---

## Cite

If you use this code or build on the idea, please cite:

```
@article{dasgupta2023alphaelimination,
  title   = {Alpha Elimination: Using Deep Reinforcement Learning to Reduce Fill-In during Sparse Matrix Decomposition},
  author  = {Arpan Dasgupta and Pawan Kumar},
  journal = {arXiv preprint arXiv:2310.09852},
  year    = {2023}
}
```

arXiv: https://arxiv.org/abs/2310.09852

---

## License

This project is for research and educational use. If you add a license file (e.g., MIT/BSD/GPL), mention it here.

---

### Acknowledgments

We thank the sparse linear algebra and RL communities for open tools and datasets, including SuiteSparse and standard sparse ordering baselines referenced above.
