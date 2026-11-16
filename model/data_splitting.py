"""Dataset partitioning, saved-fold loading, leakage checks, and fold summaries."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

from .data_loading import (
    GroupKFold,
    SKLEARN_SPLIT_AVAILABLE,
    StratifiedGroupKFold,
    build_beta_only_node_tables,
    clean_seq,
    ensure_dir,
    load_graph_split,
    now,
    read_csv_required,
    require_columns,
)

class DisjointSet:
    """Small deterministic union-find used to keep beta-sequence components intact."""

    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        self.add(x)
        parent = self.parent[x]
        if parent != x:
            self.parent[x] = self.find(parent)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        # Lexicographic root choice keeps component identifiers deterministic.
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def load_supervision_pool(nodes_dir: str) -> pd.DataFrame:
    """Merge the original train/val/test binding edges into one evaluation pool."""
    frames: List[pd.DataFrame] = []
    for split_name in ("train", "val", "test"):
        frame = load_graph_split(nodes_dir, split_name).copy()
        frame["original_split"] = split_name
        frame["original_row"] = np.arange(len(frame), dtype=int)
        frames.append(frame)
    pool = pd.concat(frames, ignore_index=True, sort=False)
    require_columns(pool, ["src", "dst", "label"], "combined supervision pool")
    pool["src"] = pool["src"].astype(str)
    pool["dst"] = pool["dst"].astype(str)
    pool["label"] = pool["label"].astype(float)
    pool["supervision_row_id"] = np.arange(len(pool), dtype=int)
    return pool


def attach_donor_metadata(pool: pd.DataFrame, nodes_dir: str) -> pd.DataFrame:
    """Resolve donor per TCR instance for LODO without using donor as a model feature."""
    out = pool.copy()
    inst_path = os.path.join(nodes_dir, "tcr_instance_nodes.csv")
    inst_df = read_csv_required(inst_path).copy()
    require_columns(inst_df, ["node_id"], inst_path)
    inst_df["node_id"] = inst_df["node_id"].astype(str)

    donor_map: Dict[str, str] = {}
    if "donor" in inst_df.columns:
        donor_values = inst_df[["node_id", "donor"]].copy()
        donor_values["donor"] = donor_values["donor"].astype(str)
        donor_values = donor_values[~donor_values["donor"].isin(["", "nan", "None"])]
        donor_map = dict(zip(donor_values["node_id"], donor_values["donor"]))

    if "donor" not in out.columns:
        out["donor"] = out["src"].map(donor_map)
    else:
        current = out["donor"].astype(str)
        missing = current.isin(["", "nan", "None", "NaN"])
        if missing.any():
            out.loc[missing, "donor"] = out.loc[missing, "src"].map(donor_map)

    if out["donor"].isna().any() or out["donor"].astype(str).isin(["", "nan", "None"]).any():
        missing_instances = out.loc[
            out["donor"].isna() | out["donor"].astype(str).isin(["", "nan", "None"]),
            "src",
        ].astype(str).drop_duplicates().head(10).tolist()
        raise ValueError(
            "Could not resolve donor for all supervised TCR instances. "
            f"Missing examples={missing_instances}. Provide a donor column in "
            "train/val/test_edges.csv or tcr_instance_nodes.csv."
        )
    out["donor"] = out["donor"].astype(str)
    return out


def build_beta_component_groups(
    nodes_dir: str,
    beta_node_key: str,
    supervised_instances: Sequence[str],
    strict_cdr3b: bool = False,
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Group beta-sequence <-> instance components for grouped evaluation.

    Strict mode additionally joins rebuilt beta nodes with identical raw CDR3B,
    irrespective of their V/J background. This changes the split groups only;
    the graph's V/J-aware sequence nodes and model inputs remain unchanged.
    """
    (
        _node_table,
        _tables,
        _node_id_to_idx,
        _node_type_to_id,
        original_to_beta,
        beta_mapping_audit,
        _beta_vdj_edges,
    ) = build_beta_only_node_tables(nodes_dir, beta_node_key)

    seqinst_path = os.path.join(nodes_dir, "tcr_sequence_to_instance_edges.csv")
    seqinst_raw = read_csv_required(seqinst_path).copy()
    require_columns(seqinst_raw, ["src", "dst"], seqinst_path)
    seqinst_raw = seqinst_raw[["src", "dst"]].astype(str).drop_duplicates()
    seqinst_raw["beta_src"] = seqinst_raw["src"].map(original_to_beta)
    if seqinst_raw["beta_src"].isna().any():
        examples = seqinst_raw.loc[seqinst_raw["beta_src"].isna(), "src"].head(10).tolist()
        raise ValueError(
            "Cannot remap some sequence-instance edges to rebuilt beta nodes; "
            f"examples={examples}"
        )

    dsu = DisjointSet()
    beta_by_instance: Dict[str, List[str]] = {}
    for beta_id, inst_id in seqinst_raw[["beta_src", "dst"]].itertuples(index=False):
        beta_id = str(beta_id)
        inst_id = str(inst_id)
        b = f"B:{beta_id}"
        i = f"I:{inst_id}"
        dsu.union(b, i)
        beta_by_instance.setdefault(inst_id, []).append(beta_id)

    if strict_cdr3b:
        # An instance may have multiple beta nodes. Keep its whole graph
        # component together while joining all V/J variants of each raw CDR3B.
        for beta_id, raw_beta in beta_mapping_audit[
            ["beta_sequence_id", "cdr3b"]
        ].drop_duplicates().itertuples(index=False, name=None):
            dsu.union(f"B:{beta_id}", f"S:{clean_seq(raw_beta)}")

    supervised_set = set(map(str, supervised_instances))
    missing = sorted(supervised_set - set(beta_by_instance))
    if missing:
        raise ValueError(
            "Some supervised TCR instances have no beta-sequence mapping; "
            f"examples={missing[:10]}"
        )

    root_to_instances: Dict[str, List[str]] = {}
    for inst_id in sorted(supervised_set):
        root = dsu.find(f"I:{inst_id}")
        root_to_instances.setdefault(root, []).append(inst_id)

    sorted_roots = sorted(root_to_instances)
    root_to_group = {
        root: f"beta_component_{idx + 1:06d}"
        for idx, root in enumerate(sorted_roots)
    }
    instance_to_group = {
        inst_id: root_to_group[root]
        for root, instances in root_to_instances.items()
        for inst_id in instances
    }

    component_beta_nodes: Dict[str, List[str]] = {}
    for inst_id in sorted(supervised_set):
        group = instance_to_group[inst_id]
        component_beta_nodes.setdefault(group, []).extend(beta_by_instance[inst_id])
    component_beta_nodes = {
        group: sorted(set(beta_ids))
        for group, beta_ids in component_beta_nodes.items()
    }
    return instance_to_group, component_beta_nodes


def prepare_validation_pool(
    args, strict_cdr3b: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    pool = load_supervision_pool(args.nodes_dir)
    pool = attach_donor_metadata(pool, args.nodes_dir)
    instance_to_group, component_beta_nodes = build_beta_component_groups(
        args.nodes_dir,
        args.beta_node_key,
        pool["src"].astype(str).unique().tolist(),
        strict_cdr3b=strict_cdr3b,
    )
    pool["beta_component_group"] = pool["src"].astype(str).map(instance_to_group)
    if pool["beta_component_group"].isna().any():
        raise RuntimeError("Internal beta-component assignment failed")
    return pool, component_beta_nodes


def _donor_label_strata(frame: pd.DataFrame) -> np.ndarray:
    """Build donor-by-label strata for donor-balanced grouped splitting.

    The beta-component group remains the indivisible unit.  Donor and label are
    used only as the stratification target so that every fold approaches the
    global donor/positive-negative composition as closely as the component
    constraints permit.
    """
    require_columns(frame, ["donor", "label"], "donor-balanced grouped split")
    donor = frame["donor"].astype(str)
    missing_donor = donor.isin(["", "nan", "None", "NaN"])
    if missing_donor.any():
        donor = donor.mask(missing_donor, "__missing_donor__")
    labels = frame["label"].astype(int).astype(str)
    return (donor + "__label__" + labels).to_numpy(dtype=object)


def _subset_distribution_error(
    frame: pd.DataFrame,
    subset_indices: np.ndarray,
    target_fraction: float,
) -> float:
    """Score size, donor, and donor-by-label imbalance for one subset."""
    subset_indices = np.asarray(subset_indices, dtype=int)
    if len(frame) == 0:
        return float("inf")

    subset = frame.iloc[subset_indices]
    observed_fraction = len(subset) / float(len(frame))
    size_error = abs(observed_fraction - float(target_fraction))

    all_donors = sorted(frame["donor"].astype(str).unique().tolist())
    global_donor = (
        frame["donor"].astype(str).value_counts(normalize=True)
        .reindex(all_donors, fill_value=0.0)
    )
    subset_donor = (
        subset["donor"].astype(str).value_counts(normalize=True)
        .reindex(all_donors, fill_value=0.0)
    )
    donor_error = 0.5 * float(np.abs(subset_donor.to_numpy() - global_donor.to_numpy()).sum())

    global_strata_values = _donor_label_strata(frame)
    subset_strata_values = global_strata_values[subset_indices]
    all_strata = sorted(set(global_strata_values.tolist()))
    global_strata = (
        pd.Series(global_strata_values).value_counts(normalize=True)
        .reindex(all_strata, fill_value=0.0)
    )
    subset_strata = (
        pd.Series(subset_strata_values).value_counts(normalize=True)
        .reindex(all_strata, fill_value=0.0)
    )
    strata_error = 0.5 * float(
        np.abs(subset_strata.to_numpy() - global_strata.to_numpy()).sum()
    )

    present_donors = set(subset["donor"].astype(str).unique().tolist())
    missing_donors = len(set(all_donors) - present_donors)
    missing_penalty = 0.25 * missing_donors / max(len(all_donors), 1)

    # Size remains part of the objective, while donor and donor-by-label
    # composition receive equal additional weight.  The missing-donor penalty
    # makes the splitter prefer folds containing every donor whenever the
    # indivisible beta components allow it.
    return float(size_error + donor_error + strata_error + missing_penalty)


def _partition_balance_score(
    frame: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
    target_fraction: float,
) -> float:
    """Score a complete grouped partition by its worst and mean fold balance."""
    scores = [
        _subset_distribution_error(frame, test_indices, target_fraction)
        for _train_indices, test_indices in splits
    ]
    if not scores:
        return float("inf")
    return float(np.mean(scores) + 0.5 * np.max(scores))


def _make_donor_balanced_group_folds(
    frame: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create deterministic donor/label-balanced, beta-component-grouped folds.

    StratifiedGroupKFold already enforces group integrity.  Multiple seeded
    candidates are evaluated and the most balanced partition is retained.  This
    improves donor representation without ever splitting a beta component.
    """
    if StratifiedGroupKFold is None:
        raise RuntimeError(
            "StratifiedGroupKFold is required for donor-balanced grouped CV; "
            "upgrade scikit-learn or use the original grouped splitter explicitly."
        )

    groups = frame["beta_component_group"].astype(str).to_numpy()
    strata = _donor_label_strata(frame)
    n_trials = max(8, min(32, n_splits * 8))
    best_splits: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
    best_score = float("inf")
    best_trial = -1

    for trial in range(n_trials):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=int(seed) + trial,
        )
        candidate = list(
            splitter.split(np.zeros(len(frame)), strata, groups)
        )
        score = _partition_balance_score(
            frame,
            candidate,
            target_fraction=1.0 / float(n_splits),
        )
        if score < best_score:
            best_score = score
            best_splits = candidate
            best_trial = trial

    if best_splits is None:
        raise RuntimeError("Could not construct donor-balanced grouped folds")

    print(
        f"[{now()}] Donor-balanced grouped split: n_splits={n_splits} "
        f"strata=donor+label trials={n_trials} selected_trial={best_trial} "
        f"balance_score={best_score:.6f}"
    )
    return best_splits


def make_group_outer_folds(
    frame: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if not SKLEARN_SPLIT_AVAILABLE:
        raise RuntimeError("scikit-learn model_selection is required for grouped CV")
    groups = frame["beta_component_group"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} beta-component groups; found {len(unique_groups)}"
        )

    if StratifiedGroupKFold is not None and len(np.unique(_donor_label_strata(frame))) > 1:
        return _make_donor_balanced_group_folds(frame, n_splits, seed)

    # This fallback preserves the original grouped behavior when stratified
    # grouped splitting is unavailable or the frame has only one stratum.
    labels = frame["label"].astype(int).to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    print(
        f"[{now()}] WARNING: donor-balanced stratification unavailable; "
        "falling back to GroupKFold."
    )
    return list(splitter.split(np.zeros(len(frame)), labels, groups))


def make_group_train_val_split(
    frame: pd.DataFrame,
    val_fraction: float,
    seed: int,
    donor_balanced: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create beta-component-disjoint train/validation sets.

    The original label-stratified grouped split is retained by default.  The
    group-5 protocol opts into donor+label balancing explicitly; this keeps the
    LODO protocol's original inner-split behavior unchanged.
    """
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be between 0 and 0.5")
    groups = frame["beta_component_group"].astype(str).to_numpy()
    labels = frame["label"].astype(int).to_numpy()
    unique_groups = np.unique(groups)
    desired_splits = max(2, int(round(1.0 / val_fraction)))
    n_splits = min(desired_splits, len(unique_groups))
    if n_splits < 2:
        raise ValueError("Not enough beta-component groups to construct train/val split")

    if donor_balanced and StratifiedGroupKFold is not None and len(np.unique(_donor_label_strata(frame))) > 1:
        candidates = _make_donor_balanced_group_folds(frame, n_splits, seed)
    elif StratifiedGroupKFold is not None and len(np.unique(labels)) > 1:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )
        candidates = list(splitter.split(np.zeros(len(frame)), labels, groups))
    else:
        splitter = GroupKFold(n_splits=n_splits)
        if donor_balanced:
            print(
                f"[{now()}] WARNING: donor-balanced stratification unavailable; "
                "falling back to GroupKFold for train/val split."
            )
        candidates = list(splitter.split(np.zeros(len(frame)), labels, groups))

    if donor_balanced:
        # Choose the candidate whose validation fraction and donor/donor-label
        # composition are jointly closest to the development-pool targets.
        best = min(
            candidates,
            key=lambda pair: _subset_distribution_error(frame, pair[1], val_fraction),
        )
    else:
        # Preserve the original behavior for protocols that did not opt into
        # donor balancing: choose the closest validation-set size.
        best = min(
            candidates,
            key=lambda pair: abs(len(pair[1]) / max(len(frame), 1) - val_fraction),
        )
    return np.asarray(best[0]), np.asarray(best[1])


def summarize_split_frame(frame: pd.DataFrame) -> Dict[str, object]:
    labels = frame["label"].astype(int)
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    donors = sorted(frame["donor"].astype(str).unique().tolist()) if "donor" in frame.columns else []
    donor_edge_counts: Dict[str, int] = {}
    donor_positive_counts: Dict[str, int] = {}
    donor_negative_counts: Dict[str, int] = {}
    donor_positive_fractions: Dict[str, float] = {}
    if "donor" in frame.columns:
        donor_values = frame["donor"].astype(str)
        donor_edge_counts = {
            str(key): int(value)
            for key, value in donor_values.value_counts().sort_index().to_dict().items()
        }
        donor_positive_counts = {
            str(key): int(value)
            for key, value in (
                frame.loc[labels == 1, "donor"].astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            )
        }
        donor_negative_counts = {
            str(key): int(value)
            for key, value in (
                frame.loc[labels == 0, "donor"].astype(str)
                .value_counts()
                .sort_index()
                .to_dict()
                .items()
            )
        }
        donor_positive_fractions = {
            donor: float(
                donor_positive_counts.get(donor, 0)
                / max(donor_edge_counts.get(donor, 0), 1)
            )
            for donor in donors
        }
    return {
        "n_edges": int(len(frame)),
        "n_positive": positives,
        "n_negative": negatives,
        "positive_fraction": float(positives / len(frame)) if len(frame) else float("nan"),
        "n_instances": int(frame["src"].astype(str).nunique()),
        "n_peptides": int(frame["dst"].astype(str).nunique()),
        "n_beta_components": int(frame["beta_component_group"].astype(str).nunique())
        if "beta_component_group" in frame.columns else 0,
        "donors": donors,
        "n_donors": int(len(donors)),
        "donor_edge_counts": donor_edge_counts,
        "donor_positive_counts": donor_positive_counts,
        "donor_negative_counts": donor_negative_counts,
        "donor_positive_fractions": donor_positive_fractions,
    }


def validate_group_disjointness(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    require_test_disjoint: bool,
) -> Dict[str, int]:
    train_groups = set(train_df["beta_component_group"].astype(str))
    val_groups = set(val_df["beta_component_group"].astype(str))
    test_groups = set(test_df["beta_component_group"].astype(str))
    train_val_overlap = len(train_groups & val_groups)
    train_test_overlap = len(train_groups & test_groups)
    val_test_overlap = len(val_groups & test_groups)
    if train_val_overlap:
        raise ValueError(f"Train/val beta-component leakage detected: {train_val_overlap}")
    if require_test_disjoint and (train_test_overlap or val_test_overlap):
        raise ValueError(
            "Grouped-CV test leakage detected: "
            f"train-test={train_test_overlap}, val-test={val_test_overlap}"
        )
    return {
        "train_val_group_overlap": train_val_overlap,
        "train_test_group_overlap": train_test_overlap,
        "val_test_group_overlap": val_test_overlap,
    }


def validate_strict_cdr3b_disjointness(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, int]:
    """Require raw CDR3B strings to be disjoint across all three splits."""
    split_betas: Dict[str, set] = {}
    for name, frame in (("train", train_df), ("val", val_df), ("test", test_df)):
        require_columns(frame, ["cdr3b"], f"{name}_edges.csv (Strict TCR)")
        sequences = frame["cdr3b"].map(clean_seq)
        if sequences.eq("").any():
            raise ValueError(f"Strict TCR requires nonempty CDR3B in {name}_edges.csv")
        split_betas[name] = set(sequences)

    overlaps = {
        "train_val_raw_cdr3b_overlap": len(split_betas["train"] & split_betas["val"]),
        "train_test_raw_cdr3b_overlap": len(split_betas["train"] & split_betas["test"]),
        "val_test_raw_cdr3b_overlap": len(split_betas["val"] & split_betas["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(
            "Strict TCR split requires disjoint raw CDR3B sequences: "
            f"{overlaps}. Generate fresh five-fold files with "
            "--group5_split_key raw_cdr3b and without "
            "--reuse_existing_five_fold_data."
        )
    return overlaps


def save_split_audit(
    fold_dir: str,
    split_frames: Dict[str, pd.DataFrame],
    split_metadata: Dict[str, object],
    split_dir_name: str = "splits",
) -> None:
    split_dir = (
        fold_dir
        if not split_dir_name
        else os.path.join(fold_dir, split_dir_name)
    )
    ensure_dir(split_dir)
    for split_name in ("train", "val", "test"):
        split_frames[split_name].to_csv(
            os.path.join(split_dir, f"{split_name}_edges.csv"),
            index=False,
        )
    payload = dict(split_metadata)
    payload["train"] = summarize_split_frame(split_frames["train"])
    payload["val"] = summarize_split_frame(split_frames["val"])
    payload["test"] = summarize_split_frame(split_frames["test"])
    with open(os.path.join(split_dir, "split_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_existing_group5_fold(
    five_fold_data_root: str,
    fold_index: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    """Load one previously saved group-5 split without recreating it."""
    fold_dir = os.path.join(
        os.path.abspath(five_fold_data_root),
        f"fold{fold_index}",
    )
    if not os.path.isdir(fold_dir):
        raise FileNotFoundError(
            f"Existing five-fold directory does not exist: {fold_dir}"
        )

    split_frames: Dict[str, pd.DataFrame] = {}
    for split_name in ("train", "val", "test"):
        split_path = os.path.join(fold_dir, f"{split_name}_edges.csv")
        frame = read_csv_required(split_path).copy().reset_index(drop=True)
        require_columns(frame, ["src", "dst", "label"], split_path)
        frame["src"] = frame["src"].astype(str)
        frame["dst"] = frame["dst"].astype(str)
        frame["label"] = pd.to_numeric(
            frame["label"], errors="raise"
        ).astype(float)
        split_frames[split_name] = frame

    split_metadata: Dict[str, object] = {
        "protocol": "group5_cv",
        "fold": f"{fold_index}fold",
        "data_fold": f"fold{fold_index}",
        "existing_split_data": True,
        "source_fold_directory": fold_dir,
    }
    summary_path = os.path.join(fold_dir, "split_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                source_summary = json.load(handle)
            if isinstance(source_summary, dict):
                for key in (
                    "outer_test_definition",
                    "donor_used_for_splitting",
                    "donor_used_for_stratification",
                    "donor_stratification_target",
                    "inner_validation_grouped_by_beta_component",
                    "train_val_group_overlap",
                    "train_test_group_overlap",
                    "val_test_group_overlap",
                ):
                    if key in source_summary:
                        split_metadata[key] = source_summary[key]
        except (OSError, json.JSONDecodeError) as exc:
            print(
                f"[{now()}] WARNING: could not read existing split summary "
                f"{summary_path}: {exc}"
            )
    return split_frames, split_metadata


def add_protocol_columns(
    frame: pd.DataFrame,
    protocol: str,
    fold_name: str,
    test_donor: str = "",
) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "protocol", protocol)
    result.insert(1, "fold", fold_name)
    result.insert(2, "test_donor", test_donor)
    return result


def aggregate_fold_metrics(fold_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    if fold_df.empty:
        result = pd.DataFrame()
        result.to_csv(output_path, index=False)
        return result
    group_cols = [
        "protocol",
        "encoder_mode_canonical",
        "graph_mode_canonical",
        "gate_label",
    ]
    metric_cols = [
        "test_auc",
        "test_aupr",
        "test_acc@0.5",
        "test_f1@0.5",
        "test_precision@0.5",
        "test_recall@0.5",
        "test_mcc@0.5",
        "test_f1@best",
        "test_mcc@best",
    ]
    metric_cols = [column for column in metric_cols if column in fold_df.columns]
    rows: List[Dict[str, object]] = []
    for keys, group in fold_df.groupby(group_cols, dropna=False):
        row: Dict[str, object] = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row["n_folds"] = int(len(group))
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().astype(float)
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_min"] = float(values.min()) if len(values) else float("nan")
            row[f"{metric}_max"] = float(values.max()) if len(values) else float("nan")
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, index=False)
    return result
