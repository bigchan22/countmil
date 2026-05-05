# Remote Experiment Plan

This repo can run the same manifest launcher on each server. Keep every server's outputs under a unique tag so no result files collide when you copy them back.

## Setup

On each remote server:

```bash
git clone git@github.com:bigchan22/countmil.git
cd countmil
python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install torch torchvision numpy pytest scikit-learn
```

Copy datasets into `data/` before launching. Do not rely on the remote network unless you explicitly want to retry downloads.

Required for current MNIST/FashionMNIST jobs:

```text
data/MNIST/raw/train-images-idx3-ubyte.gz
data/MNIST/raw/train-labels-idx1-ubyte.gz
data/MNIST/raw/t10k-images-idx3-ubyte.gz
data/MNIST/raw/t10k-labels-idx1-ubyte.gz
data/FashionMNIST/raw/train-images-idx3-ubyte.gz
data/FashionMNIST/raw/train-labels-idx1-ubyte.gz
data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
data/FashionMNIST/raw/t10k-labels-idx1-ubyte.gz
```

Optional CIFAR archives for later work:

```text
data/cifar-10-python.tar.gz   md5 c58f30108f718f92721af3b95e74349a
data/cifar-100-python.tar.gz  md5 eb9058c3a382ffc7106e4002c42a8d85
```

Verify the machine before running:

```bash
nvidia-smi
PYTHONPATH=src .venv/bin/python -m pytest
```

## Server Assignments

Current 4x4090 server:

- Finish `configs/neurips_pilot/preleave_20260504_1619/manifest.tsv`.
- Aggregate results and inspect failures before starting any major reruns.

3090 server:

- Run MNIST expected-sum baselines and histogram LLP baselines. These are the most important comparison jobs missing from the paper story.

```bash
TAG=server3090_$(date -u +%Y%m%d_%H%M)
PYTHONPATH=src .venv/bin/python scripts/launcher/make_digit_sum_baseline_grid.py --tag "$TAG"
PYTHONPATH=src .venv/bin/python scripts/launcher/make_histogram_llp_grid.py --tag "$TAG"
tmux new-session -d -s countmil_3090_digit_sum_baselines "GPUS='0 1 2 3' EPOCHS=50 BATCH_SIZE=128 TEST_BAGS=1000 RUN_ROOT=runs/server3090_digit_sum_baselines RESULTS_ROOT=results/server3090_digit_sum_baselines LOG_DIR=logs/server3090_digit_sum_baselines PYTHONPATH=src scripts/launcher/run_neurips_manifest.sh configs/digit_sum_baseline/$TAG/manifest.tsv"
tmux new-session -d -s countmil_3090_histogram_llp "GPUS='0 1 2 3' EPOCHS=50 BATCH_SIZE=128 TEST_BAGS=1000 RUN_ROOT=runs/server3090_histogram_llp RESULTS_ROOT=results/server3090_histogram_llp LOG_DIR=logs/server3090_histogram_llp PYTHONPATH=src scripts/launcher/run_neurips_manifest.sh configs/histogram_llp/$TAG/manifest.tsv"
```

A5000 server:

- Run shorter robustness jobs. Treat it as fragile: smaller batch, fewer epochs first, then scale only if stable.

```bash
TAG=serverA5000_$(date -u +%Y%m%d_%H%M)
PYTHONPATH=src .venv/bin/python scripts/launcher/make_fashionmnist_grid.py --tag "$TAG" --train-bags 1000 --seeds 0 1 --bag-settings 10:2 50:10
tmux new-session -d -s countmil_a5000_fashion "GPUS='0 1 2 3' EPOCHS=30 BATCH_SIZE=96 TEST_BAGS=1000 RUN_ROOT=runs/serverA5000_fashion RESULTS_ROOT=results/serverA5000_fashion LOG_DIR=logs/serverA5000_fashion PYTHONPATH=src scripts/launcher/run_neurips_manifest.sh configs/fashionmnist/$TAG/manifest.tsv"
```

## Monitoring

```bash
tmux ls
tmux capture-pane -pt countmil_3090_digit_sum_baselines -S -80
nvidia-smi
PYTHONPATH=src .venv/bin/python scripts/aggregate_results.py --input results/server3090_digit_sum_baselines --output results/server3090_digit_sum_baselines/aggregate.csv
```

If a server dies, do not delete partial outputs. Generate a remaining-job manifest from the summaries that are missing, or ask Codex to make one.
