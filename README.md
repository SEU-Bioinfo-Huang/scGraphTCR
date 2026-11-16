# scGraphTCR

![scGraphTCR graphical abstract](assets/graphical-abstract.png)

We developed scGraphTCR to predict which peptides a T-cell receptor (TCR) binds. It combines pretrained encoders for CDR3β and peptide sequences with a directed graph that incorporates V/J gene usage and links receptors to T-cell states and pathway activities inferred from single-cell transcriptomes. scGraphTCR outperformed seven baselines in five-fold CDR3β-disjoint cross-validation. Evaluation in an independent single-cell cohort supported its potential for prioritizing candidate peptides.

This repository provides the processed internal graph, fixed splits, training code and checkpoints required to reproduce the five-fold internal benchmark.

## Installation

Run the commands from the repository root. The study used Python 3.10.20 and PyTorch 2.4.1 with CUDA 12.4.

```bash
conda env create -f environment.yml
conda activate scgraphtcr
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
```

For CPU evaluation, install `torch==2.4.1` from `https://download.pytorch.org/whl/cpu` and pass `--device cpu`. See [PyTorch 2.4.1 installation options](https://pytorch.org/get-started/previous-versions/#v241).

## Evaluate the saved model

```bash
python model.py \
  --nodes_dir data/graph/data \
  --five_fold_data_dir data/5_fold \
  --reuse_existing_five_fold_data \
  --validation_protocol group5 \
  --encoder_mode self_both \
  --single_mode full_no_residue \
  --graph_attention_mode learned \
  --gate_mode fixed_0.3 \
  --evaluate_only \
  --checkpoint_root train_result \
  --save_predictions \
  --outdir results/evaluation \
  --device cuda \
  --seed 50
```

This evaluates five saved checkpoints on the supplied CDR3β-disjoint folds and writes `group5_fold_metrics.csv`, `group5_mean_std.csv` and per-fold predictions. The reference [fold summary](data/result/group5_mean_std.csv) gives mean test AUROC 0.746 and mean test AUPR 0.364 for the complete model. AUPR is calculated as average precision. Hardware and software versions can affect numerical results.

## Retrain the model and sequence/graph ablations

```bash
python model.py \
  --nodes_dir data/graph/data \
  --five_fold_data_dir data/5_fold \
  --reuse_existing_five_fold_data \
  --validation_protocol group5 \
  --encoder_mode self_both \
  --run_ablation \
  --ablation_modes only_seq,only_graph,full_no_residue \
  --graph_attention_mode learned \
  --gate_mode fixed_0.3 \
  --stage2_epochs 100 \
  --freeze_encoder_epochs 100 \
  --unfreeze_mode frozen \
  --save_predictions \
  --outdir results/ablation \
  --device cuda \
  --seed 50
```

For the complete model alone, replace the two ablation options with `--single_mode full_no_residue`. The provided checkpoints cover the complete model. The sequence-only and graph-only checkpoints can be regenerated with the ablation command. Add `--save_interpretability --save_transformer_attention` to regenerate attention exports. The [recorded run configuration](data/result/run_config.json) includes the full parameter settings.

## Data and input paths

- `data/graph/data/` contains the model-ready graph, source labelled pairs, cell-state annotations and KEGG signature definitions. Use `--nodes_dir` to point to another graph.
- `data/5_fold/` contains the original five train/validation/test assignments. Use `--five_fold_data_dir` to point to another set of folds. To create new CDR3β-disjoint folds, omit `--reuse_existing_five_fold_data` and use a new fold directory.
- `pretrain_result/` contains the two encoder checkpoints used by default. Set `--self_tcr_encoder_path` and `--self_peptide_encoder_path` to use new weights. See [encoder pretraining](data/pretrain/use.md) for the sequence corpora and training command.
- `train_result/` contains the complete-model checkpoint for each fold. Set `--checkpoint_root` to evaluate a different run. Set `--outdir` to choose where to save new outputs.

### 10x expression data

The internal data come from the four healthy CD8+ T-cell donor datasets from 10x Genomics. For graph construction, download the filtered feature-barcode H5 and matching V(D)J all-contig annotation CSV for [donor 1](https://www.10xgenomics.com/datasets/cd-8-plus-t-cells-of-healthy-donor-1-1-standard-3-0-2), [donor 2](https://www.10xgenomics.com/datasets/cd-8-plus-t-cells-of-healthy-donor-2-1-standard-3-0-2), [donor 3](https://www.10xgenomics.com/datasets/cd-8-plus-t-cells-of-healthy-donor-3-1-standard-3-0-2), and [donor 4](https://www.10xgenomics.com/datasets/cd-8-plus-t-cells-of-healthy-donor-4-1-standard-3-0-2). See [graph construction](data/graph/use.md) for the input schema and paths to fill in.

### External inference

The manuscript's external cohort is GEO [GSE268157](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE268157). Reproducing the manuscript's external retrieval benchmark requires its processed candidate pairs, sample/group identifiers, operational-positive labels, and retrieval analysis. This package provides the scoring entry point. Supply a compatible candidate graph in place of `/path/to/external_graph`.

```bash
python model.py \
  --nodes_dir data/graph/data \
  --external_inference \
  --external_graph_dir /path/to/external_graph \
  --checkpoint_root train_result \
  --encoder_mode self_both \
  --single_mode full_no_residue \
  --graph_attention_mode learned \
  --gate_mode fixed_0.3 \
  --outdir results/external \
  --device cuda \
  --seed 50
```

External inference writes per-fold and ensemble candidate scores. The internal code and checkpoints cover scGraphTCR and its sequence/graph ablations. Reproducing the paper's other-method comparisons requires the corresponding comparator implementations and inputs.
