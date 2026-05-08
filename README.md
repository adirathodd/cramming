# Cramming in Multi-Term Modular Arithmetic Sequence Models

This repository contains the anonymized code release and example artifacts for the ICML submission:

**"Cramming: End-to-end trained Sequence Models Defer Computation to the Last Token"**

The project studies mechanistic interpretability and generalization in modular arithmetic sequence models, with an emphasis on:
- Fourier-spectrum structure in learned representations
- SVD- and component-ablation analyses
- Cross-length generalization failures ("cramming")
- Scratchpad/chain-of-thought supervision (`seq_cot`) as a fix

The paper PDF is included as [`Research Paper.pdf`](./Research%20Paper.pdf).

## Repository Scope

This repo includes:
- Training and analysis code for RNN and Transformer models
- Config-driven experiment workflows (`config.yaml` per run)
- Analysis modules for Fourier, SVD, trigonometric checks, and ablations
- Sample configs and a notebook under [`samples/`](./samples)
- Example model outputs and figures under [`final_models/`](./final_models)
- Paper rerun experiment bundle under [`paper_rerun_models/`](./paper_rerun_models)

## Environment Setup

### 1. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run inside the virtual environment

All commands below assume the virtual environment is activated:

```bash
source .venv/bin/activate
```

## Quickstart

### Train a model

```bash
python3 main.py -t -o addition -m transformer
```

### Analyze a trained model

```bash
python3 main.py -a -o addition -m transformer
```

### Train + analyze in one run

```bash
python3 main.py -t -a -o addition subtraction -m transformer
```

### Reuse an existing run directory

```bash
python3 main.py -t -a -p transformer_modular_addition
```

## Command-Line Interface

Main entry point: [`main.py`](./main.py)

```text
-p, --path         Run directory containing config.yaml
-o, --operation    One or more operations: addition subtraction multiplication division
-m, --model-type   rnn | transformer
-t, --train        Enable training
-a, --analysis     Enable analysis
--device           cpu | cuda | cuda:0 ...
--verbose          Print full model/config details
```

If `-p` is not provided, run directories are auto-named as:

```text
{model_type}_modular_{sorted_ops}
```

## Configuration-First Workflow

Each run is controlled by a `config.yaml` in its run directory.

Recommended loop:
1. Create or edit `<run_dir>/config.yaml`.
2. Train with `-t`.
3. Analyze with `-a`.
4. Inspect figures under `<run_dir>/figures/`.
5. Iterate on settings (`nterms`, `training_target`, `batch_mode`, Fourier regularization, etc.).

Key config modules:
- [`config/main.py`](./config/main.py)
- [`config/data.py`](./config/data.py)
- [`config/training.py`](./config/training.py)
- [`config/analysis.py`](./config/analysis.py)

## Data / Task Format

The dataset uses modular arithmetic sequences in the form:

```text
[a1, op, a2, op, ..., an, =]
```

- Sequence length is determined by `nterms` (or `nterms_list` for variable-length settings).
- Operations can be single-op or multi-op.
- Supervision target is controlled by `training_target`:
  - `last_token`: supervise final answer only
  - `seq_cot`: supervise intermediate scratchpad states + final answer

## Cramming-Relevant Experimental Settings

To reproduce the core behavior discussed in the paper:
- Use Transformer runs with `training_target: last_token` for cramming behavior.
- Evaluate cross-length generalization (`m != n`) via analysis outputs.
- Compare against `training_target: seq_cot` to observe improved generalization and non-deferred computation.

Important compatibility checks implemented in config validation:
- `training_target='seq_cot'` with transformer requires `model.mask=True`
- `batch_mode` must be one of `full`, `mini-batch`, `operation-batch`
- Mixed additive/multiplicative Fourier-regularized runs require `operation-batch`

## Checkpoints and Outputs

Per run directory:
- `best.pt`: checkpoint at best test performance (grokking state)
- `final.pt`: checkpoint from final epoch
- `figures/`: generated Fourier/SVD/ablation/generalization plots
- `config.yaml`: exact experiment config used

Analysis mode loads `best.pt` by default and falls back to `final.pt` if needed.

## Tests

```bash
python3 tests/test_datasets.py
python3 tests/test_fourier_losses.py
```

## Included Examples

- Starter configs and notebook: [`samples/`](./samples)
- Example completed runs and figures: [`final_models/`](./final_models)
- Paper-focused rerun configs/results: [`paper_rerun_models/`](./paper_rerun_models)

A practical starting point is:
- [`samples/configs/transformer_addition_quickstart.yaml`](./samples/configs/transformer_addition_quickstart.yaml)
- [`samples/configs/transformer_multiop_seqcot_sample.yaml`](./samples/configs/transformer_multiop_seqcot_sample.yaml)

## `paper_rerun_models` Guide

`paper_rerun_models/` is the primary directory for paper reproducibility runs. Each run folder should contain:
- `config.yaml`
- `checkpoints/` (for example `best.pt`, `final.pt`)
- `figures/` (analysis outputs)

Folder conventions:
- `cot/` means scratchpad/chain-of-thought supervision (`training_target: seq_cot`)
- `mask/` or `e2e-mask/` means end-to-end style runs (final-answer supervision, i.e. `last_token`)

Example paths:
- `paper_rerun_models/multi-op/3_term/cot/add_sub`
- `paper_rerun_models/multi-op/3_term/mask/add_sub`
- `paper_rerun_models/single-op/4_term/cot/add`

## Notebook Usage

The main analysis notebooks are:
- [`notebooks/analysis.ipynb`](./notebooks/analysis.ipynb)
- [`notebooks/prefix_and_nterms_generalization_analysis.ipynb`](./notebooks/prefix_and_nterms_generalization_analysis.ipynb)

Recommended usage:
1. Train/analyze from CLI or Slurm into a run directory under `paper_rerun_models/`.
2. Open one of the notebooks and select the run directory using the top configuration/picker cell.
3. Re-run all cells to regenerate metrics/plots for that run.

For cluster training, see scripts in [`scripts/`](./scripts), including:
- [`scripts/run_transformer_multiop_2term_cot.slurm.sh`](./scripts/run_transformer_multiop_2term_cot.slurm.sh)

## Project Structure

```text
analysis/      Fourier, SVD, ablation, trig, variable-length metrics
config/        Validated config schemas and defaults
datasets/      Modular arithmetic dataset generation
losses/        Fourier regularization losses (modes 1-7)
models/        RNN and Transformer implementations
training/      Training loop, batching, evaluation utilities
samples/       Reproducible starter configs + notebook
final_models/  Example trained run artifacts and generated figures
paper_rerun_models/  Paper reproducibility runs (configs/checkpoints/figures)
notebooks/     Interactive analysis notebooks for run inspection
```

## Anonymization Note

This release is intentionally anonymized for peer review. Identifying metadata has been removed or replaced.

## Citation

If this code is useful in your work, please cite the associated paper (camera-ready citation to be added after review).
