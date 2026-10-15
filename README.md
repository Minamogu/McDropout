# PneumoniaMNIST MC Dropout experiment

This implementation runs on a normal Linux Python environment and does not
use Google Colab APIs. It follows `proposals/20261007.md`: the CNN, early
stopping, five seeds, three repeats per seed, 30 MC Dropout passes, ECE,
predictive entropy, mutual information, error-detection AUROC, and plots are
included.

## Setup on the GPU server

Use the CUDA-specific PyTorch wheel recommended for the server before
installing the remaining packages. Then install this project's dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The first execution creates the data directory and downloads PneumoniaMNIST
into `data/`. The directory must be writable by the account running the job;
use an absolute path such as `/home/mina/medmnist-data` if the current
directory is not writable. The original
MedMNIST splits contain slightly more samples than the plan requests; this
script uses the first 4,700/500/600 samples from train/validation/test,
respectively, so every run uses exactly the planned sizes.

## Full experiment

```bash
python train_mc_dropout.py --data-dir data --output-dir results --device cuda
```

The default is 5 seeds × 3 repeats × 100 epochs (with early stopping).

## One-run result check

Before launching the full experiment, run one seed and one training run with:

```bash
python run_single_experiment.py \
    --data-dir /home/mina/medmnist-data \
    --output-dir /home/mina/mcdropout-single \
    --device cuda \
    --seed 42
```

This writes one `metrics.csv`, `history.csv`, checkpoint, MC prediction file,
and plots to the specified output directory. It uses the same model and
evaluation logic as the full experiment, but does not create the multi-seed
summary files.

The current default regularization settings are `weight_decay=1e-4`,
`patience=4`, and `min_delta=0.001`. The dropout rate and CNN architecture
remain unchanged from the experiment plan.

For a short smoke run:

```bash
python run_single_experiment.py --device cpu --epochs 1 --seed 42 --skip-plots
```

Outputs include:

- `run_metrics.csv`: every seed/repeat result
- `seed_representatives.csv`: mean and sample standard deviation over repeats
- `experiment_summary.csv`: mean, variance, and standard deviation over seed representatives
- per-run `history.csv`, `model.pt`, `mc_predictions.npz`, and PNG plots

The MC Dropout helper puts the whole model in `eval()` and then enables only
Dropout modules, leaving BatchNorm in evaluation mode as required.

Progress is printed immediately to standard output (`flush=True`). During
training, each line shows the current epoch, training/validation loss and
accuracy, best validation loss, and the early-stopping patience counter.
Progress is also shown for deterministic test evaluation and MC Dropout
passes. To keep a server log while viewing the output, use:

```bash
python train_mc_dropout.py --device cuda 2>&1 | tee experiment.log
```
