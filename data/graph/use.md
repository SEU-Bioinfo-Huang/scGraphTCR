# Graph Construction

`build_complete_tcr_graph.py` converts TCR–peptide pairs, 10x expression H5 files, and V(D)J annotations into the node tables, relation tables, and supervised pair files required by the model. It can also generate four-state T-cell annotations and UCell pathway scores.

## Input Preparation

Pair CSVs require six columns: `donor, barcode, cdr3a, cdr3b, peptide, label`, with binary labels of 0 or 1. In inference-only mode, `label` may be omitted, but `cdr3a` must still be nonempty. The current script accepts only `donor1`–`donor4`. Preserve complete barcode suffixes such as `-1` and `-2`.

Place expression H5 files and V(D)J annotation files in separate directories. When running from the repository root, the default directories are `inputs/10x/expression` and `inputs/10x/vdj`. The default filenames for donor1 are:

```text
vdj_v1_hs_aggregated_donor1_filtered_feature_bc_matrix.h5
vdj_v1_hs_aggregated_donor1_all_contig_annotations.csv
```

Use the corresponding donor numbers for donor2–donor4; files are needed only for donors present in the input pairs. For other filenames, specify `--h5 donor1=/path/to/file.h5` and `--vdj-contig donor1=/path/to/file.csv`. H5 download links are provided in the [repository README](../../README.md#10x-expression-data).

In addition to the model environment's dependencies, including NumPy and pandas, graph construction uses:

```bash
pip install scanpy anndata harmonypy scipy h5py pyucell requests urllib3 igraph leidenalg scikit-misc
```

Run the following Bash commands from the repository root. First, replace the input paths:

```bash
H5_DIR=/path/to/expression_h5
VDJ_DIR=/path/to/vdj_annotations
```

## 1. Existing Train/Validation/Test Pair Files

Supply three CSVs with the six columns above, in train/validation/test order. Their assignments are preserved. The script rejects pairs or paired CDR3α/β sequences that occur across multiple partitions.

```bash
python data/graph/build_complete_tcr_graph.py \
  --split-files /path/to/train.csv /path/to/val.csv /path/to/test.csv \
  --h5-dir "$H5_DIR" \
  --vdj-dir "$VDJ_DIR" \
  --outdir /path/to/new_graph
```

## 2. A Single Labelled Pair File

If the CSV contains a `split` column, its values must be `train`, `val`, or `test`. Otherwise, the script creates a deterministic split grouped by paired CDR3α/β sequences, targeting 70%/15%/15% with seed 42 by default. Actual proportions depend on the sequence groups.

```bash
python data/graph/build_complete_tcr_graph.py \
  --binding-file /path/to/labelled_pairs.csv \
  --h5-dir "$H5_DIR" \
  --vdj-dir "$VDJ_DIR" \
  --outdir /path/to/new_graph
```

## 3. Unlabelled Candidate Pairs

Only the first five columns are required. `--inference-only` places all candidates in `test_edges.csv` with empty labels. This step constructs the graph without producing model predictions. The donor1–donor4 restriction still applies.

```bash
python data/graph/build_complete_tcr_graph.py \
  --binding-file /path/to/unlabelled_pairs.csv \
  --inference-only \
  --h5-dir "$H5_DIR" \
  --vdj-dir "$VDJ_DIR" \
  --outdir /path/to/new_inference_graph
```

To obtain candidate-pair scores, subsequently run the model's `--external_inference` mode with this graph, the original training graph, and the corresponding checkpoints.

## Reusing Annotations and Reading Outputs

To rebuild the original graph with the supplied pair partitions, state annotations and fixed pathway gene sets, run:

```bash
python data/graph/build_complete_tcr_graph.py \
  --split-files data/graph/data/train_edges.csv data/graph/data/val_edges.csv data/graph/data/test_edges.csv \
  --state-annotations data/graph/data/selected_cell_4state_annotations.csv \
  --kegg-signatures-json data/graph/data/kegg_pathway_signatures.json \
  --h5-dir "$H5_DIR" \
  --vdj-dir "$VDJ_DIR" \
  --outdir results/rebuilt_graph
```

The original graph-build log records Scanpy 1.11.5, AnnData 0.11.4, HarmonyPy 2.0.0 and pyUCell 0.7.3. Expression H5 files remain required for UCell scoring. With no annotation file or signature cache, the script recomputes state annotations and retrieves KEGG gene sets online. For a different cell population, review the state annotations and supply your own annotation file or a reviewed mapping through `--cluster-state-map`.

All outputs are written to `--outdir`, including node/relation CSVs, `train_edges.csv`, `val_edges.csv`, `test_edges.csv`, and graph validation summaries. Pass this directory to the model through `--nodes_dir`. These partitions are distinct from the model's strict CDR3β-disjoint five-fold assignments, which are generated or loaded by the model script.

The examples use new output directories. Add `--overwrite` only when replacing existing graph files. Run `python data/graph/build_complete_tcr_graph.py --help` for additional options.
