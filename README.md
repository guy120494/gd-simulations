
# Thesis Experiments — Gradient Descent Dynamics

This repository contains code for running controlled gradient descent experiments on a 1D two-neuron ReLU network, plus wider two-layer ReLU networks on a disk dataset.

The main CLI (`python main.py`) runs **Experiment 5f**. Trainable-output-weight runs (A–D) and disk experiments are documented below.

---

# Model

We study a 1D ReLU network with two neurons:

[
f(x) = v_1 \mathrm{ReLU}(w_1 x + b_1) + v_2 \mathrm{ReLU}(w_2 x + b_2)
]

For this experiment:

* Number of neurons: `k = 2`
* Output weights: `v = [1, -1]` (fixed during training)
* Initialization:

  * ( w_1, w_2 \sim \mathcal{N}(0,2) )
  * ( b_1 = b_2 = 0 )

Training is performed using full-batch gradient descent with **exponential loss**:

[
L = \frac{1}{n} \sum_i \exp(-y_i f(x_i))
]

---

# Experiment — Description

Each run:

1. Initializes the network randomly.
2. Runs gradient descent.
3. Stops when one of the following occurs:

### Hit Condition

Both:

* ( |w_1 + b_1 + w_2 - b_2| < \text{tol} )
* Loss < `loss_threshold`

### Loss Abort

If after 10,000 iterations the loss is still above the threshold.

### Max Iterations

If `max_iterations` is reached.

The experiment records:

* Hit times
* Final parameters
* Loss values
* A geometric metric:
  [
  \min(|b_2 - b_1|,\ |w_1^{(0)} + w_2^{(0)}|/2)
  ]

---

# How to Run

From the project root directory:

```bash
python main.py
```

By default this runs:

* `num_runs = 10_000`
* `learning_rate = 0.01`
* `max_iterations = 10_000_000`
* `seed = 42`

---

# CLI Arguments

You can override parameters:

```bash
python main.py \
  --runs 10000 \
  --lr 0.05 \
  --max-iterations 1000000 \
  --seed 123
```

### Available Arguments

| Argument                    | Description                                  |
| --------------------------- | -------------------------------------------- |
| `--runs`                    | Number of independent random initializations |
| `--lr` or `--learning-rate` | Learning rate                                |
| `--max-iterations`          | Maximum GD iterations                        |
| `--seed` or `-s`            | Random seed                                  |

---

# Output Files

Running the experiment produces:

* `experiment_5f_runs.csv` — detailed results per run
* `experiment_5f_summary.txt` — overall statistics
* `experiment_5f_hit_time_hist.png`
* `experiment_5f_hit_time_hist.csv`
* `experiment_5f_metric_hist.png`
* `experiment_5f_metric_hist.csv`

---

# Experiment — Trainable Output Weights

Same 1D two-neuron ReLU network and exponential loss as Experiment 5f. Data are the two points ( (-1, -1) ) and ( (1, 1) ). Training is simultaneous full-batch gradient descent in float64, with a constant learning rate and no momentum or regularization.

Each run:

1. Initializes the network from one of the cohorts below.
2. Runs gradient descent in one mode:
   * Fixed: only ( w, b ) are trained. ( v ) stays at its initial value.
   * Trainable: all six parameters are trained with the same learning rate.
3. Stops when one of the following occurs:

### Loss Abort

If at step 10,000 the loss is still at least `0.5`. Status code `1`.

### Nonfinite

If a parameter becomes non-finite. Status code `2`.

### Max Iterations

If the step budget for that run is reached. Status code `0` (completed).

The experiment records, at geometrically spaced checkpoints and at each decade of steps:

* Parameters ( w_1, b_1, w_2, b_2, v_1, v_2 )
* Loss
* Margin: exact distance from the nearest training point to the zero set
* Gap: ( 1 - \text{margin} )
* Status and the step at which the run stopped

### Initialization

Two cohorts. Both start with ( b_1 = b_2 = 0 ) and ( v = (1, -1) ), unless a run overrides ( v ).

* He: ( w_1, w_2 \sim \mathcal{N}(0, 2) ), `default_rng(42)`, the same draws as the paper's main experiment
* Theorem box: 32 draws with ( w_1 \sim U(2.7, 3) ) and ( w_2 \sim -U(0, 0.5) ), `default_rng(42)`

Run C also uses these output-weight initializations (trainable mode only):

* `unit`: ( v = (1, -1) )
* `balanced`: ( |v_j| = \|(w_j, b_j)\| ), signs ( +, - )
* `random`: ( v \sim \mathcal{N}(0, 1) )
* `small`: ( v = (0.1, -0.1) )

### Runs

| Run | What is trained                                                                 | Steps | Learning rate |
| --- | ------------------------------------------------------------------------------- | ----- | ------------- |
| A   | 10,000 He + 32 box, fixed and trainable                                        | 10^7  | 0.05          |
| B   | 200 He seeds that completed both modes in A (seed 2026) + 32 box, both modes   | 10^9  | 0.05          |
| C   | first 2,000 He + 32 box, trainable only, four ( v ) initializations            | 10^7  | 0.05          |
| D   | first 2,000 He + 32 box, fixed and trainable                                   | 10^7  | 0.01 and 0.1  |

Run B reads `results_A/trajectories.npz` and keeps He seeds whose fixed and trainable runs both completed.

---

# How to Run — Trainable Output Weights

From the directory that contains `run_experiment.py` (needs Python 3 with numpy, and a C++17 compiler for `train_kernel.cpp`):

```bash
python run_experiment.py A
```

Then, in order:

```bash
python run_experiment.py B
python run_experiment.py C
python run_experiment.py D
python analyze.py
```

`python run_experiment.py test` builds the kernel and runs the self-tests only.

---

# CLI Arguments — Trainable Output Weights

```bash
python run_experiment.py A --workers 12
```

| Argument   | Description                                                                 |
| ---------- | --------------------------------------------------------------------------- |
| `exp`      | `test`, `A`, `B`, `C`, `D`, or `calibrate`                                  |
| `--workers`| Thread pool size for training. Default: `os.cpu_count()`                    |

`calibrate` times a single run of ( 10^8 ) steps and prints steps per second.

---

# Output Files — Trainable Output Weights

Each run writes a directory `results_A`, `results_B`, `results_C`, or `results_D`:

* `trajectories.npz` — checkpoints, parameters, loss, margin, gap, status, stop step, and the run config
* `runs.csv` — one row per run: config, status, initial and final parameters, final loss and margin, gap at each decade from ( 10^3 ) onward
* `checkpoints_decades.csv` — completed runs at ( t = 0 ) and at each decade
* `meta.json` — step budget, run counts, timings, machine, and validation checks

`python analyze.py` reads those directories and writes `summary.json` and the figures under `figures/`.

---

# Wide Network Experiments

`main_disks.py` trains a 2-layer with 10 neurons ReLU network on the 2-disk dataset (`wide_network_experiments/experiment_disks.py`).

The commands used to run the disk experiments (200 runs, lr=0.1, number of neurons=10, input's dimension=5, training set size=20):

```bash
# Adam
python main_disks.py --optimizer adam --max-iterations 1000001 --k 10 --d 5 --n 20 --lr 0.1 --num-runs 200 --report-every 100000

# GD
python main_disks.py --optimizer gd --max-iterations 1000001 --k 10 --d 5 --n 20 --lr 0.1 --num-runs 200 --report-every 100000
```

See `python main_disks.py --help` for the full list of arguments (dataset size, network width, optimizer, output paths, etc.).

---

# Project Structure

```
core.py           # Network, gradients, GD
experiments.py    # Experiment 5f implementation
datasets.py       # Dataset generation
init_utils.py     # Parameter initialization
cli.py            # Command-line interface
main.py           # Entry point

wide_network_experiments/disk_*.py    # Disk experiment code (dataset, network, trainer, CLI)
main_disks.py                   # Entry point — run this to run the disk experiments
run_experiment.py / analyze.py  # Trainable-output-weight runs A–D (see section above)
```

---

# Notes

* Output weights `v` are fixed.
