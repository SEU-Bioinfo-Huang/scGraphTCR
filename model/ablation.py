

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import pandas as pd
import torch

from .data_loading import (
    DEFAULT_FIVE_FOLD_DATA_DIR,
    SKLEARN_SPLIT_AVAILABLE,
    SelfTransformerEncoderBundle,
    encoder_mode_settings,
    ensure_dir,
    load_graph_split,
    load_self_transformer_bundle,
    mode_settings,
    now,
    parse_csv_list,
    resolve_sequence_lengths,
    set_seed,
)
from .data_splitting import (
    add_protocol_columns,
    aggregate_fold_metrics,
    load_existing_group5_fold,
    make_group_outer_folds,
    make_group_train_val_split,
    prepare_validation_pool,
    save_split_audit,
    validate_group_disjointness,
    validate_strict_cdr3b_disjointness,
)
from .training import (
    parse_gate_configuration,
    run_external_inference,
    train_one_ablation,
)

def run_model_grid(
    args,
    self_bundle: SelfTransformerEncoderBundle,
    device: torch.device,
    root_outdir: str,
    split_frames: Dict[str, pd.DataFrame],
    split_metadata: Dict[str, object],
) -> pd.DataFrame:
    """Run the exact existing encoder/graph/gate grid on one externally supplied split."""
    ensure_dir(root_outdir)
    if args.run_encoder_ablation:
        encoder_modes = parse_csv_list(args.encoder_modes)
    else:
        encoder_modes = [args.encoder_mode]

    if args.run_ablation:
        graph_modes = parse_csv_list(args.ablation_modes)
    else:
        graph_modes = [args.single_mode]

    if args.run_gate_ablation:
        requested_gate_modes = parse_csv_list(args.gate_modes)
    else:
        requested_gate_modes = [args.gate_mode]
    requested_gate_modes = [
        parse_gate_configuration(spec, args.gate_init).requested
        for spec in requested_gate_modes
    ]
    use_gate_subdir = bool(args.run_gate_ablation or len(requested_gate_modes) > 1)

    all_summaries: List[Dict[str, object]] = []
    for encoder_mode in encoder_modes:
        encoder_settings = encoder_mode_settings(encoder_mode)
        canonical_encoder_mode = str(encoder_settings["canonical_mode"])
        encoder_outdir = os.path.join(root_outdir, canonical_encoder_mode)
        ensure_dir(encoder_outdir)
        print(
            f"[{now()}] Encoder experiment={canonical_encoder_mode}; "
            f"pretrained_tcr={encoder_settings['pretrained_tcr']} "
            f"pretrained_peptide={encoder_settings['pretrained_peptide']}"
        )
        encoder_summaries: List[Dict[str, object]] = []
        for graph_mode in graph_modes:
            canonical_graph_mode = str(mode_settings(graph_mode)["canonical_mode"])
            gate_modes_for_graph = (
                ["fixed_0.0"]
                if canonical_graph_mode in {"only_seq", "no_graph"}
                else requested_gate_modes
            )
            for gate_mode in gate_modes_for_graph:
                summary = train_one_ablation(
                    args=args,
                    mode=graph_mode,
                    gate_spec=gate_mode,
                    self_bundle=self_bundle,
                    encoder_mode=canonical_encoder_mode,
                    device=device,
                    root_outdir=encoder_outdir,
                    use_gate_subdir=use_gate_subdir,
                    split_frames=split_frames,
                    split_metadata=split_metadata,
                )
                encoder_summaries.append(summary)
                all_summaries.append(summary)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        pd.DataFrame(encoder_summaries).to_csv(
            os.path.join(encoder_outdir, "ablation_summary.csv"),
            index=False,
        )

    combined = pd.DataFrame(all_summaries)
    combined.to_csv(
        os.path.join(root_outdir, "encoder_graph_gate_ablation_summary.csv"),
        index=False,
    )
    combined.to_csv(
        os.path.join(root_outdir, "encoder_graph_ablation_summary.csv"),
        index=False,
    )
    return combined


def run_existing_group5_protocol(
    args,
    self_bundle: SelfTransformerEncoderBundle,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate or train on previously saved group-5 split files."""
    root = os.path.abspath(args.outdir)
    ensure_dir(root)
    fold_summaries: List[pd.DataFrame] = []

    for fold_index in range(1, args.group_cv_folds + 1):
        fold_name = f"{fold_index}fold"
        fold_dir = os.path.join(root, fold_name)
        split_frames, split_metadata = load_existing_group5_fold(
            args.five_fold_data_dir,
            fold_index,
        )
        if args.group5_split_key == "raw_cdr3b":
            overlap = validate_group_disjointness(
                split_frames["train"], split_frames["val"],
                split_frames["test"], require_test_disjoint=True,
            )
            overlap.update(validate_strict_cdr3b_disjointness(
                split_frames["train"], split_frames["val"], split_frames["test"],
            ))
            split_metadata.update(overlap)
            split_metadata["outer_test_definition"] = "raw_cdr3b_strict"
            split_metadata["inner_validation_grouped_by_raw_cdr3b"] = True
        split_metadata.update({
            "protocol": "group5_cv",
            "fold": fold_name,
            "data_fold": f"fold{fold_index}",
            "existing_split_data": True,
        })
        save_split_audit(fold_dir, split_frames, split_metadata)
        print(
            f"[{now()}] GROUP5 existing {fold_name}: "
            f"train={len(split_frames['train'])} "
            f"val={len(split_frames['val'])} "
            f"test={len(split_frames['test'])} "
            f"data_dir={os.path.join(os.path.abspath(args.five_fold_data_dir), f'fold{fold_index}')}"
        )
        fold_summary = run_model_grid(
            args,
            self_bundle,
            device,
            root_outdir=os.path.join(fold_dir, "models"),
            split_frames=split_frames,
            split_metadata=split_metadata,
        )
        fold_summary = add_protocol_columns(fold_summary, "group5_cv", fold_name)
        fold_summaries.append(fold_summary)

    if not fold_summaries:
        raise RuntimeError("No existing five-fold split was loaded.")
    combined = pd.concat(fold_summaries, ignore_index=True, sort=False)
    combined.to_csv(os.path.join(root, "group5_fold_metrics.csv"), index=False)
    aggregate_fold_metrics(combined, os.path.join(root, "group5_mean_std.csv"))
    return combined


def run_group5_protocol(
    args,
    pool: pd.DataFrame,
    self_bundle: SelfTransformerEncoderBundle,
    device: torch.device,
) -> pd.DataFrame:
    if args.reuse_existing_five_fold_data:
        return run_existing_group5_protocol(args, self_bundle, device)

    root = os.path.join(args.outdir)
    ensure_dir(root)
    five_fold_data_root = os.path.abspath(args.five_fold_data_dir)
    if args.group5_split_key == "raw_cdr3b":
        existing = [
            f"fold{i}" for i in range(1, args.group_cv_folds + 1)
            if any(os.path.exists(os.path.join(
                five_fold_data_root, f"fold{i}", f"{name}_edges.csv",
            )) for name in ("train", "val", "test"))
        ]
        if existing:
            raise FileExistsError(
                "Strict TCR split needs a fresh --five_fold_data_dir; existing "
                f"fold files would be overwritten: {existing}"
            )
        prior_results = [
            f"{i}fold" for i in range(1, args.group_cv_folds + 1)
            if os.path.exists(os.path.join(root, f"{i}fold"))
        ]
        if prior_results:
            raise FileExistsError(
                "Strict TCR training needs a fresh --outdir; existing fold "
                f"results would be overwritten: {prior_results}"
            )
    ensure_dir(five_fold_data_root)
    outer_folds = make_group_outer_folds(pool, args.group_cv_folds, args.seed)
    fold_summaries: List[pd.DataFrame] = []

    for fold_index, (dev_idx, test_idx) in enumerate(outer_folds, start=1):
        fold_name = f"{fold_index}fold"
        fold_dir = os.path.join(root, fold_name)
        fold_data_dir = os.path.join(five_fold_data_root, f"fold{fold_index}")
        dev_df = pool.iloc[dev_idx].reset_index(drop=True)
        test_df = pool.iloc[test_idx].reset_index(drop=True)
        inner_train_idx, inner_val_idx = make_group_train_val_split(
            dev_df,
            args.cv_val_fraction,
            args.seed + 1000 + fold_index,
            donor_balanced=True,
        )
        train_df = dev_df.iloc[inner_train_idx].reset_index(drop=True)
        val_df = dev_df.iloc[inner_val_idx].reset_index(drop=True)
        split_frames = {"train": train_df, "val": val_df, "test": test_df}
        overlap = validate_group_disjointness(
            train_df, val_df, test_df, require_test_disjoint=True
        )
        if args.group5_split_key == "raw_cdr3b":
            overlap.update(validate_strict_cdr3b_disjointness(
                train_df, val_df, test_df,
            ))
        split_metadata: Dict[str, object] = {
            "protocol": "group5_cv",
            "fold": fold_name,
            "data_fold": f"fold{fold_index}",
            "outer_test_definition": (
                "raw_cdr3b_strict" if args.group5_split_key == "raw_cdr3b"
                else "beta_sequence_instance_connected_component"
            ),
            "donor_used_for_splitting": False,
            "donor_used_for_stratification": True,
            "donor_stratification_target": "donor_and_label",
            "inner_validation_grouped_by_beta_component": True,
            "inner_validation_grouped_by_raw_cdr3b": args.group5_split_key == "raw_cdr3b",
            **overlap,
        }
        save_split_audit(fold_dir, split_frames, split_metadata)
        save_split_audit(
            fold_data_dir,
            split_frames,
            split_metadata,
            split_dir_name="",
        )
        print(
            f"[{now()}] GROUP5 {fold_name}: "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
            f"train_groups={train_df['beta_component_group'].nunique()} "
            f"val_groups={val_df['beta_component_group'].nunique()} "
            f"test_groups={test_df['beta_component_group'].nunique()} "
            f"train_donors={sorted(train_df['donor'].astype(str).unique().tolist())} "
            f"val_donors={sorted(val_df['donor'].astype(str).unique().tolist())} "
            f"test_donors={sorted(test_df['donor'].astype(str).unique().tolist())} "
            f"data_dir={fold_data_dir}"
        )
        fold_summary = run_model_grid(
            args,
            self_bundle,
            device,
            root_outdir=os.path.join(fold_dir, "models"),
            split_frames=split_frames,
            split_metadata=split_metadata,
        )
        fold_summary = add_protocol_columns(fold_summary, "group5_cv", fold_name)
        fold_summaries.append(fold_summary)

    combined = pd.concat(fold_summaries, ignore_index=True, sort=False)
    combined.to_csv(os.path.join(root, "group5_fold_metrics.csv"), index=False)
    aggregate_fold_metrics(combined, os.path.join(root, "group5_mean_std.csv"))
    return combined


def run_lodo_protocol(
    args,
    pool: pd.DataFrame,
    self_bundle: SelfTransformerEncoderBundle,
    device: torch.device,
) -> pd.DataFrame:
    root = os.path.join(args.outdir, "lodo")
    ensure_dir(root)
    requested_donors = parse_csv_list(args.lodo_donors)
    observed_donors = sorted(pool["donor"].astype(str).unique().tolist())
    if not requested_donors:
        requested_donors = observed_donors
    missing = [donor for donor in requested_donors if donor not in observed_donors]
    if missing:
        raise ValueError(
            f"Requested LODO donors not found: {missing}; observed={observed_donors}"
        )

    fold_summaries: List[pd.DataFrame] = []
    for fold_index, test_donor in enumerate(requested_donors, start=1):
        fold_name = f"test_{test_donor}"
        fold_dir = os.path.join(root, fold_name)
        test_mask = pool["donor"].astype(str).eq(test_donor).to_numpy()
        test_df = pool.loc[test_mask].reset_index(drop=True)
        dev_df = pool.loc[~test_mask].reset_index(drop=True)
        if test_df.empty or dev_df.empty:
            raise ValueError(f"Invalid LODO split for donor={test_donor}")

        inner_train_idx, inner_val_idx = make_group_train_val_split(
            dev_df,
            args.cv_val_fraction,
            args.seed + 2000 + fold_index,
        )
        train_df = dev_df.iloc[inner_train_idx].reset_index(drop=True)
        val_df = dev_df.iloc[inner_val_idx].reset_index(drop=True)
        split_frames = {"train": train_df, "val": val_df, "test": test_df}
        overlap = validate_group_disjointness(
            train_df, val_df, test_df, require_test_disjoint=False
        )
        train_donors = sorted(train_df["donor"].astype(str).unique().tolist())
        val_donors = sorted(val_df["donor"].astype(str).unique().tolist())
        if test_donor in train_donors or test_donor in val_donors:
            raise RuntimeError(f"LODO donor leakage detected for {test_donor}")

        # Sequence overlap is reported, not forbidden: LODO tests unseen donor,
        # while grouped CV separately tests unseen beta-sequence components.
        split_metadata = {
            "protocol": "lodo",
            "fold": fold_name,
            "test_donor": test_donor,
            "outer_test_definition": "entire_donor",
            "donor_used_for_splitting": True,
            "inner_validation_grouped_by_beta_component": True,
            **overlap,
        }
        save_split_audit(fold_dir, split_frames, split_metadata)
        print(
            f"[{now()}] LODO {fold_name}: "
            f"train={len(train_df)} val={len(val_df)} test={len(test_df)} "
            f"train_donors={train_donors} val_donors={val_donors} "
            f"test_donor={test_donor} "
            f"train_test_beta_component_overlap={overlap['train_test_group_overlap']}"
        )
        fold_summary = run_model_grid(
            args,
            self_bundle,
            device,
            root_outdir=os.path.join(fold_dir, "models"),
            split_frames=split_frames,
            split_metadata=split_metadata,
        )
        fold_summary = add_protocol_columns(
            fold_summary,
            "lodo",
            fold_name,
            test_donor=test_donor,
        )
        fold_summaries.append(fold_summary)

    combined = pd.concat(fold_summaries, ignore_index=True, sort=False)
    combined.to_csv(os.path.join(root, "lodo_fold_metrics.csv"), index=False)
    aggregate_fold_metrics(combined, os.path.join(root, "lodo_mean_std.csv"))
    return combined


def run_original_protocol(
    args,
    self_bundle: SelfTransformerEncoderBundle,
    device: torch.device,
) -> pd.DataFrame:
    split_frames = {
        name: load_graph_split(args.nodes_dir, name)
        for name in ("train", "val", "test")
    }
    metadata = {
        "protocol": "original",
        "fold": "original_split",
        "outer_test_definition": "existing_train_val_test_files",
    }
    root = os.path.join(args.outdir, "original_split")
    save_split_audit(root, split_frames, metadata)
    combined = run_model_grid(
        args,
        self_bundle,
        device,
        root_outdir=os.path.join(root, "models"),
        split_frames=split_frames,
        split_metadata=metadata,
    )
    return add_protocol_columns(combined, "original", "original_split")


def parse_args() -> argparse.Namespace:
    pretrained_dir = Path(__file__).resolve().parents[1] / "pretrain_result"
    parser = argparse.ArgumentParser(
        description=(
            "Load self-pretrained Transformer checkpoints and train a CDR3B-only "
            "directed Stage-2 context graph predictor"
        )
    )

    # Paths
    parser.add_argument("--nodes_dir", type=str, required=True)
    parser.add_argument(
        "--self_tcr_encoder_path",
        type=str,
        default=str(pretrained_dir / "self_tcr_transformer_encoder.pt"),
        help="TCR Transformer checkpoint. Defaults to the supplied pretrained encoder.",
    )
    parser.add_argument(
        "--self_peptide_encoder_path",
        type=str,
        default=str(pretrained_dir / "self_peptide_transformer_encoder.pt"),
        help="Peptide Transformer checkpoint. Defaults to the supplied pretrained encoder.",
    )
    parser.add_argument(
        "--teinet_encoder_dir",
        dest="self_encoder_dir_compat",
        type=str,
        default="",
        help="Deprecated compatibility argument; ignored by this self-pretrained version.",
    )
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument(
        "--five_fold_data_dir",
        type=str,
        default=DEFAULT_FIVE_FOLD_DATA_DIR,
        help=(
            "Directory for grouped 5-fold data. It is written during a new split; "
            "with --reuse_existing_five_fold_data it is read as fold1, fold2, ... ."
        ),
    )
    parser.add_argument(
        "--reuse_existing_five_fold_data",
        action="store_true",
        help=(
            "Read existing fold1..foldN train_edges.csv, val_edges.csv, and "
            "test_edges.csv instead of recreating the grouped splits."
        ),
    )
    parser.add_argument(
        "--group5_split_key",
        choices=["raw_cdr3b", "beta_component"],
        default="raw_cdr3b",
        help=(
            "Group5 fold identity: raw_cdr3b (Strict TCR, default) joins all "
            "V/J variants of the same CDR3B; beta_component reproduces the "
            "previous V/J-aware component split. Graph nodes are unchanged."
        ),
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help=(
            "Load existing Stage-2 checkpoints and evaluate/export results "
            "without training. Use with --reuse_existing_five_fold_data."
        ),
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default="",
        help=(
            "Root of previous fold results containing "
            "{1fold,...}/models/{encoder}/{mode}/stage2_best.pt."
        ),
    )
    parser.add_argument(
        "--external_inference",
        action="store_true",
        help=(
            "Run the saved final model on an externally constructed graph and "
            "write probabilities without training or label-based metrics."
        ),
    )
    parser.add_argument(
        "--external_graph_dir",
        "--external_nodes_dir",
        dest="external_graph_dir",
        type=str,
        default="",
        help=(
            "External graph directory produced by build_complete_tcr_graph.py. "
            "Use together with --external_inference."
        ),
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=114514)
    parser.add_argument(
        "--seq_dir",
        type=str,
        default="",
        help="Compatibility-only argument; self-pretrained checkpoints are loaded directly and no new sequence pretraining is run.",
    )
    parser.add_argument(
        "--beta_node_key",
        type=str,
        default="cdr3b_vj",
        choices=["cdr3b", "cdr3b_vj"],
        help=(
            "How original alpha-beta sequence nodes are collapsed: cdr3b_vj keeps "
            "distinct beta V/J backgrounds; cdr3b merges all nodes sharing CDR3B."
        ),
    )

    # Self-pretrained Transformer input/encoder configuration
    parser.add_argument(
        "--max_cdr3_len",
        type=int,
        default=0,
        help="CDR3B tensor length. Use 0 to inherit the TCR checkpoint positional length.",
    )
    parser.add_argument(
        "--max_peptide_len",
        type=int,
        default=0,
        help="Peptide tensor length. Use 0 to inherit the peptide checkpoint positional length.",
    )
    parser.add_argument(
        "--self_n_heads",
        type=int,
        default=4,
        help="Fallback attention-head count when checkpoint metadata does not store it.",
    )
    parser.add_argument(
        "--self_encoder_dropout",
        "--teinet_encoder_dropout",
        dest="self_encoder_dropout",
        type=float,
        default=0.10,
        help=(
            "Fallback Transformer dropout when checkpoint metadata does not store it; "
            "the self-pretraining default is 0.10."
        ),
    )
    parser.add_argument(
        "--encoder_mode",
        type=str,
        default="self_both",
        choices=[
            "self_random", "self_tcr_only", "self_peptide_only", "self_both",
            "teinet_random", "teinet_tcr_only", "teinet_epi_only", "teinet_both",
        ],
    )
    parser.add_argument(
        "--encoder_modes",
        type=str,
        default="self_random,self_tcr_only,self_peptide_only,self_both",
        help="Comma-separated modes used when --run_encoder_ablation is set.",
    )
    parser.add_argument("--run_encoder_ablation", action="store_true")

    # Directed graph bottleneck
    parser.add_argument("--graph_dim", type=int, default=64)
    parser.add_argument(
        "--graph_layers", type=int, default=1,
        help="Compatibility argument; directed hierarchy uses exactly one context step.",
    )
    parser.add_argument("--bottleneck_dropout", type=float, default=0.30)
    parser.add_argument("--graph_dropout", type=float, default=0.20)
    parser.add_argument("--decoder_dropout", type=float, default=0.25)
    parser.add_argument(
        "--graph_attention_mode",
        type=str,
        default="mean",
        choices=["mean", "learned"],
        help=(
            "mean preserves the original arithmetic graph aggregation; learned "
            "enables TCR-neighbor, TCR-VDJ, and TCR-state relation attention."
        ),
    )

# Deprecated command-line compatibility options. The pair-specific residue graph
# is removed from this final model; these options are accepted but ignored so
# older launch commands do not fail at argument parsing.
    parser.add_argument(
        "--disable_residue_graph",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--residue_scale",
        type=float,
        default=0.30,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--residue_dropout",
        type=float,
        default=0.20,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--residue_pair_batch_size",
        type=int,
        default=512,
        help=argparse.SUPPRESS,
    )

    # Graph gate configuration
    parser.add_argument(
        "--gate_mode",
        type=str,
        default="learnable",
        help=(
            "Single-run gate specification: learnable, learnable_<init>, "
            "or fixed_<value>."
        ),
    )
    parser.add_argument(
        "--gate_modes",
        type=str,
        default="fixed_0.0,fixed_0.1,fixed_0.3,fixed_0.5,fixed_1.0,learnable",
        help="Comma-separated gate specifications used with --run_gate_ablation.",
    )
    parser.add_argument(
        "--run_gate_ablation",
        action="store_true",
        help=(
            "Run each selected graph mode with every gate in --gate_modes. "
            "The only_seq/no_graph baselines are run once because their graph "
            "gates are inactive."
        ),
    )
    parser.add_argument(
        "--gate_init",
        type=float,
        default=0.10,
        help="Initial value for --gate_mode learnable and bare learnable entries.",
    )
    parser.add_argument(
        "--gate_lr_multiplier",
        type=float,
        default=20.0,
        help="Learnable gate LR = graph_lr * gate_lr_multiplier.",
    )
    parser.add_argument("--node_encode_batch_size", type=int, default=256)
    parser.add_argument(
        "--max_context_in_neighbors", type=int, default=16,
        help="Deprecated compatibility argument; reverse hub aggregation is disabled.",
    )
    parser.add_argument(
        "--include_positive_train_binding_edges", action="store_true",
        help="Rejected because peptide nodes must bypass graph propagation.",
    )

    # Optional regularization/auxiliary supervision
    parser.add_argument("--tcr_feature_mask_rate", type=float, default=0.00)
    parser.add_argument("--peptide_feature_mask_rate", type=float, default=0.00)
    parser.add_argument("--lambda_vdj", type=float, default=0.00)
    parser.add_argument("--lambda_state", type=float, default=0.00)
    parser.add_argument("--aux_edge_batch_size", type=int, default=8192)

    # Stage-2 optimization
    parser.add_argument("--stage2_epochs", type=int, default=50)
    parser.add_argument(
        "--freeze_encoder_epochs",
        type=int,
        default=50,
        help="First diagnostic should freeze self-pretrained encoders for all Stage-2 epochs.",
    )
    parser.add_argument(
        "--unfreeze_mode",
        type=str,
        default="frozen",
        choices=["frozen", "last_transformer", "last_gru", "all"],
        help="Encoder trainability after freeze_encoder_epochs.",
    )
    parser.add_argument("--graph_lr", type=float, default=1e-3)
    parser.add_argument("--encoder_finetune_lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=5.0, help="Use <=0 to estimate from train labels.")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--stage2_early_stop", type=int, default=12)
    parser.add_argument("--stage2_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")

    # Evaluation protocols. "both" runs leakage-safe beta-sequence grouped
    # 5-fold CV and 4-donor LODO in the same invocation.
    parser.add_argument(
        "--validation_protocol",
        type=str,
        default="both",
        choices=["original", "group5", "lodo", "both"],
        help=(
            "original=use existing train/val/test files; "
            "group5=donor/label-stratified beta-sequence grouped 5-fold CV; "
            "lodo=leave-one-donor-out; both=run group5 and lodo."
        ),
    )
    parser.add_argument(
        "--group_cv_folds",
        type=int,
        default=5,
        help="Number of outer beta-sequence grouped CV folds.",
    )
    parser.add_argument(
        "--cv_val_fraction",
        type=float,
        default=0.20,
        help=(
            "Validation fraction taken from each outer development pool using "
            "beta-component-disjoint grouping. For group5 this is 20%% of the "
            "remaining 80%%, i.e. about 16%% of the total pool. Donor and label "
            "composition are balanced whenever the component constraints allow."
        ),
    )
    parser.add_argument(
        "--lodo_donors",
        type=str,
        default="donor1,donor2,donor3,donor4",
        help="Comma-separated donors used as the four LODO outer test folds.",
    )

    # Graph ablations
    parser.add_argument(
        "--ablation_modes",
        type=str,
        default="only_seq,only_graph,full_no_residue",
        help=(
            "Comma-separated ablations: only_seq, only_graph, full_no_residue. "
            "The historical full_residue name is accepted as a no-residue alias."
        ),
    )
    parser.add_argument("--run_ablation", action="store_true")
    parser.add_argument("--single_mode", type=str, default="full_no_residue")
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument(
        "--save_interpretability",
        action="store_true",
        help=(
            "Export graph-attention edge tables for the selected interpretability mode."
        ),
    )
    parser.add_argument(
        "--save_transformer_attention",
        action="store_true",
        help=(
            "Expose and export self-Transformer attention for TCR and peptide "
            "sequence nodes. This is post-training diagnostics and does not "
            "change the ordinary forward output."
        ),
    )
    parser.add_argument(
        "--interpretability_mode",
        type=str,
        default="full_no_residue",
        help="Graph mode whose predictions are used for interpretability export.",
    )
    parser.add_argument(
        "--interpretability_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Split used for graph/Transformer attention analysis.",
    )
    parser.add_argument(
        "--interpretability_top_k",
        type=int,
        default=100,
        help="Number of highest-attention sequences whose Transformer self-attention maps are exported.",
    )
    parser.add_argument(
        "--interpretability_high_quantile",
        type=float,
        default=0.90,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name, path in [
        ("self TCR encoder", args.self_tcr_encoder_path),
        ("self peptide encoder", args.self_peptide_encoder_path),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} checkpoint does not exist: {path}")
    if args.self_n_heads <= 0:
        raise ValueError("--self_n_heads must be positive")
    if not 0.0 <= args.self_encoder_dropout < 1.0:
        raise ValueError("--self_encoder_dropout must be in [0,1)")
    if args.beta_node_key not in {"cdr3b", "cdr3b_vj"}:
        raise ValueError("--beta_node_key must be cdr3b or cdr3b_vj")
    if args.graph_dim <= 0:
        raise ValueError("--graph_dim must be positive")
    if args.interpretability_top_k < 0:
        raise ValueError("--interpretability_top_k must be non-negative")
    # Validate the requested target early, even when the current run only uses
    # another mode and interpretability export will be skipped for that mode.
    mode_settings(args.interpretability_mode)
    if args.group_cv_folds < 2:
        raise ValueError("--group_cv_folds must be at least 2")
    if not 0.0 < args.cv_val_fraction < 0.5:
        raise ValueError("--cv_val_fraction must be between 0 and 0.5")
    if args.reuse_existing_five_fold_data and args.validation_protocol != "group5":
        raise ValueError(
            "--reuse_existing_five_fold_data currently requires "
            "--validation_protocol group5."
        )
    if (
        args.group5_split_key == "raw_cdr3b"
        and args.validation_protocol in {"group5", "both"}
        and not args.reuse_existing_five_fold_data
    ):
        old_folds = [
            i for i in range(1, args.group_cv_folds + 1)
            if os.path.exists(os.path.join(args.outdir, f"{i}fold"))
            or any(os.path.exists(os.path.join(
                args.five_fold_data_dir, f"fold{i}", f"{split}_edges.csv",
            )) for split in ("train", "val", "test"))
        ]
        if old_folds:
            raise FileExistsError(
                "Strict TCR requires fresh --outdir and --five_fold_data_dir "
                f"to preserve old results; existing folds={old_folds}"
            )
    if args.evaluate_only:
        if not args.reuse_existing_five_fold_data:
            raise ValueError(
                "--evaluate_only requires --reuse_existing_five_fold_data."
            )
        if args.validation_protocol != "group5":
            raise ValueError(
                "--evaluate_only currently supports only "
                "--validation_protocol group5."
            )
        if not str(args.checkpoint_root).strip():
            raise ValueError(
                "--checkpoint_root is required when --evaluate_only is enabled."
            )
        if not os.path.isdir(os.path.abspath(args.checkpoint_root)):
            raise FileNotFoundError(
                f"Checkpoint root does not exist: {args.checkpoint_root}"
            )
    if args.external_inference:
        if not str(args.external_graph_dir).strip():
            raise ValueError(
                "--external_graph_dir is required when --external_inference is enabled."
            )
        if not os.path.isdir(os.path.abspath(args.external_graph_dir)):
            raise FileNotFoundError(
                f"External graph directory does not exist: {args.external_graph_dir}"
            )
        if not str(args.checkpoint_root).strip():
            raise ValueError(
                "--checkpoint_root is required when --external_inference is enabled."
            )
        if not os.path.isdir(os.path.abspath(args.checkpoint_root)):
            raise FileNotFoundError(
                f"Checkpoint root does not exist: {args.checkpoint_root}"
            )
        if args.run_ablation or args.run_encoder_ablation or args.run_gate_ablation:
            raise ValueError(
                "External inference runs one saved final model ensemble; disable "
                "--run_ablation, --run_encoder_ablation and --run_gate_ablation."
            )
    if (
        not args.external_inference
        and args.validation_protocol in {"group5", "lodo", "both"}
        and not SKLEARN_SPLIT_AVAILABLE
    ):
        raise RuntimeError("scikit-learn model_selection is required for grouped CV/LODO")

    selected_gate_specs = (
        parse_csv_list(args.gate_modes)
        if args.run_gate_ablation
        else [args.gate_mode]
    )
    if not selected_gate_specs:
        raise ValueError("At least one gate specification is required")
    for gate_spec in selected_gate_specs:
        parse_gate_configuration(gate_spec, args.gate_init)
    if not 0.0 < args.gate_init < 1.0:
        raise ValueError("--gate_init must be strictly between 0 and 1")
    if args.gate_lr_multiplier <= 0:
        raise ValueError("--gate_lr_multiplier must be positive")

    if args.max_cdr3_len < 0 or args.max_peptide_len < 0:
        raise ValueError("Sequence tensor lengths must be non-negative; 0 means infer from checkpoint")
    if args.graph_layers != 1:
        raise ValueError(
            "The directed-context architecture requires --graph_layers 1; "
            "repeated hub propagation is intentionally disabled."
        )
    if args.include_positive_train_binding_edges:
        raise ValueError(
            "--include_positive_train_binding_edges is incompatible with this model: "
            "peptide nodes bypass graph propagation."
        )
    if not 0.0 <= args.tcr_feature_mask_rate < 1.0:
        raise ValueError("--tcr_feature_mask_rate must be in [0,1)")
    if not 0.0 <= args.peptide_feature_mask_rate < 1.0:
        raise ValueError("--peptide_feature_mask_rate must be in [0,1)")
    if args.lambda_vdj < 0 or args.lambda_state < 0:
        raise ValueError("Auxiliary loss weights must be non-negative")
    if args.freeze_encoder_epochs < 0:
        raise ValueError("--freeze_encoder_epochs must be non-negative")
    if args.unfreeze_mode == "frozen" and args.freeze_encoder_epochs < args.stage2_epochs:
        print(
            f"[{now()}] Self-pretrained encoders remain frozen for all epochs because "
            "--unfreeze_mode=frozen."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    ensure_dir(args.outdir)
    set_seed(args.seed)

    device = torch.device(
        args.device
        if args.device != "cuda" or torch.cuda.is_available()
        else "cpu"
    )
    print(f"[{now()}] Device={device}")
    print(f"[{now()}] Self TCR encoder={os.path.abspath(args.self_tcr_encoder_path)}")
    print(f"[{now()}] Self peptide encoder={os.path.abspath(args.self_peptide_encoder_path)}")
    if args.self_encoder_dir_compat:
        print(
            f"[{now()}] Ignoring deprecated --teinet_encoder_dir="
            f"{args.self_encoder_dir_compat}"
        )
    print(
        f"[{now()}] Strict beta-only graph: CDR3A/TRAV/TRAJ disabled; "
        f"beta_node_key={args.beta_node_key}"
    )
    print(
        f"[{now()}] Final complete model=full_no_residue; "
        "pair-specific residue graph removed"
    )
    print(
        f"[{now()}] Graph attention mode={args.graph_attention_mode}; "
        f"Transformer attention export={args.save_transformer_attention}"
    )
    if args.save_interpretability or args.save_transformer_attention:
        print(
            f"[{now()}] Interpretability export: "
            f"mode={args.interpretability_mode} split={args.interpretability_split} "
            f"top_k={args.interpretability_top_k} "
            f"transformer={args.save_transformer_attention}"
        )
    print(
        f"[{now()}] Validation protocol={args.validation_protocol}; "
        f"group_cv_folds={args.group_cv_folds}; "
        f"cv_val_fraction={args.cv_val_fraction:g}; "
        f"lodo_donors={args.lodo_donors}"
    )
    if args.validation_protocol in {"group5", "both"}:
        if args.reuse_existing_five_fold_data:
            print(
                f"[{now()}] Group5 split: reusing existing fold files; "
                "no new split will be generated."
            )
        else:
            print(
                f"[{now()}] Group5 split: {args.group5_split_key} grouped with "
                "donor+label stratification; donor is balanced but not held out."
            )
        print(
            f"[{now()}] Five-fold dataset directory="
            f"{os.path.abspath(args.five_fold_data_dir)}"
        )
    if args.evaluate_only:
        print(
            f"[{now()}] Evaluation-only checkpoint root="
            f"{os.path.abspath(args.checkpoint_root)}"
        )

    self_bundle = load_self_transformer_bundle(args)
    resolve_sequence_lengths(args, self_bundle)

    with open(
        os.path.join(args.outdir, "run_config.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    if args.external_inference:
        print(
            f"[{now()}] External inference enabled; no training, split creation, "
            "or label-based evaluation will be run."
        )
        run_external_inference(args, self_bundle, device)
        return

    protocol_summaries: List[pd.DataFrame] = []

    if args.validation_protocol == "original":
        protocol_summaries.append(
            run_original_protocol(args, self_bundle, device)
        )
    elif args.reuse_existing_five_fold_data:
        protocol_summaries.append(
            run_existing_group5_protocol(args, self_bundle, device)
        )
    else:
        pool, component_beta_nodes = prepare_validation_pool(
            args,
            strict_cdr3b=(
                args.validation_protocol in {"group5", "both"}
                and args.group5_split_key == "raw_cdr3b"
            ),
        )
        pool.to_csv(
            os.path.join(args.outdir, "combined_supervision_pool.csv"),
            index=False,
        )
        with open(
            os.path.join(args.outdir, "beta_component_membership.json"),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(component_beta_nodes, handle, indent=2, ensure_ascii=False)
        print(
            f"[{now()}] Combined supervision pool: edges={len(pool)} "
            f"positive={int((pool['label'].astype(int) == 1).sum())} "
            f"negative={int((pool['label'].astype(int) == 0).sum())} "
            f"instances={pool['src'].nunique()} peptides={pool['dst'].nunique()} "
            f"beta_components={pool['beta_component_group'].nunique()} "
            f"donors={sorted(pool['donor'].astype(str).unique().tolist())}"
        )

        if args.validation_protocol in {"group5", "both"}:
            protocol_summaries.append(
                run_group5_protocol(args, pool, self_bundle, device)
            )
        if args.validation_protocol in {"lodo", "both"}:
            # A combined run keeps LODO's original V/J-aware inner grouping.
            lodo_pool = (
                prepare_validation_pool(args)[0]
                if args.validation_protocol == "both"
                and args.group5_split_key == "raw_cdr3b"
                else pool
            )
            protocol_summaries.append(
                run_lodo_protocol(args, lodo_pool, self_bundle, device)
            )

    if protocol_summaries:
        combined_all = pd.concat(protocol_summaries, ignore_index=True, sort=False)
        combined_all.to_csv(
            os.path.join(args.outdir, "all_protocol_fold_metrics.csv"),
            index=False,
        )
        result_path = os.path.join(args.outdir, "result.csv")
        combined_all.to_csv(result_path, index=False)
        aggregate_fold_metrics(
            combined_all,
            os.path.join(args.outdir, "all_protocol_mean_std.csv"),
        )
        print(
            f"[{now()}] Saved final result.csv, all protocol fold metrics, and "
            f"mean/std summaries under {args.outdir}"
        )
