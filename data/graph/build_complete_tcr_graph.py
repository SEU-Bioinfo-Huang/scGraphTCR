#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import inspect
import json
import re
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


VALID_DONORS = ("donor1", "donor2", "donor3", "donor4")
REQUIRED_BINDING_COLUMNS = (
    "donor",
    "barcode",
    "cdr3a",
    "cdr3b",
    "peptide",
    "label",
)
TCR_INSTANCE_KEY = ("donor", "barcode", "cdr3a", "cdr3b")
TCR_SEQUENCE_KEY = ("cdr3a", "cdr3b")

FOUR_STATES = (
    "CD8_Naive",
    "CD8_Memory_Transitional",
    "CD8_Cytotoxic_Effector",
    "CD8_TRM_like",
)


DEFAULT_CLUSTER_STATE_MAP = {
    "0": "CD8_Cytotoxic_Effector",
    "1": "CD8_Cytotoxic_Effector",
    "2": "CD8_Naive",
    "3": "CD8_Cytotoxic_Effector",
    "4": "CD8_TRM_like",
    "5": "CD8_Cytotoxic_Effector",
    "6": "CD8_Memory_Transitional",
    "7": "CD8_Memory_Transitional",
}


PATHWAY_CATALOG = (
    ("hsa04010", "MAPK signaling pathway"),
    ("hsa04060", "Cytokine-cytokine receptor interaction"),
    ("hsa04064", "NF-kappa B signaling pathway"),
    ("hsa04350", "TGF-beta signaling pathway"),
    ("hsa04612", "Antigen processing and presentation"),
    ("hsa04630", "JAK-STAT signaling pathway"),
    ("hsa04650", "Natural killer cell mediated cytotoxicity"),
    ("hsa04658", "Th1 and Th2 cell differentiation"),
    ("hsa04660", "T cell receptor signaling pathway"),
    ("hsa04668", "TNF signaling pathway"),
    ("hsa05235", "PD-L1 expression and PD-1 checkpoint pathway in cancer"),
)

GRAPH_OUTPUT_FILENAMES = (
    "tcr_sequence_nodes.csv",
    "tcr_instance_nodes.csv",
    "peptide_nodes.csv",
    "vdj_gene_nodes.csv",
    "tcellstate_context_nodes.csv",
    "tcr_sequence_to_instance_edges.csv",
    "tcr_sequence_to_vdj_gene_edges.csv",
    "tcr_instance_to_tcellstate_context_edges.csv",
    "train_edges.csv",
    "val_edges.csv",
    "test_edges.csv",
)



def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "na"}:
        return ""
    return text


def normalize_donor(value: object) -> str:
    text = clean_text(value).lower()
    match = re.search(r"donor\s*[_:\-.]?\s*([0-9]+)", text)
    if match:
        return f"donor{int(match.group(1))}"
    if re.fullmatch(r"[0-9]+", text):
        return f"donor{int(text)}"
    return text


def clean_sequence(value: object, field: str) -> str:
    sequence = re.sub(r"\s+", "", clean_text(value)).upper()
    if not sequence:
        return ""
    if not re.fullmatch(r"[A-Z*.-]+", sequence):
        raise ValueError(f"Illegal characters in {field}: {value!r}")
    return sequence


def clean_peptide(value: object) -> str:
    return clean_sequence(value, "peptide")


def extract_exact_10x_barcodes(value: object) -> List[str]:
    text = clean_text(value)
    if not text:
        return []
    return sorted(
        {
            match.group(0).upper()
            for match in re.finditer(
                r"[ACGTNacgtn]{10,30}(?:-\d+)?",
                text,
            )
        }
    )


def canonical_barcode(value: object, field: str = "barcode") -> str:
    candidates = extract_exact_10x_barcodes(value)
    if not candidates:
        raise ValueError(f"Cannot extract a 10x barcode from {field}: {value!r}")
    suffixed = [item for item in candidates if re.search(r"-\d+$", item)]
    selected = suffixed or candidates
    if len(selected) != 1:
        raise ValueError(
            f"{field} contains multiple possible 10x barcodes: {value!r} -> {selected}"
        )
    return selected[0]


def barcode_core(value: object) -> str:
    return re.sub(r"-\d+$", "", clean_text(value).upper())


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def make_tcr_instance_id(donor: str, barcode: str, cdr3a: str, cdr3b: str) -> str:
    raw = "||".join((donor, barcode, cdr3a, cdr3b))
    return "tcr_" + stable_hash(raw)


def make_tcr_sequence_id(cdr3a: str, cdr3b: str) -> str:
    return "tcrseq_" + stable_hash(f"cdr3a={cdr3a}|cdr3b={cdr3b}")


def make_peptide_id(peptide: str) -> str:
    return "peptide_" + stable_hash(peptide)


def safe_node_token(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", clean_text(value))
    text = re.sub(r"_+", "_", text).strip("_")
    if not text:
        raise ValueError(f"Cannot create node token from {value!r}")
    return text


def make_context_node_id(state: str, pathway_id: str, level: str) -> str:
    return (
        "tcellstate_context__"
        f"{safe_node_token(state)}__{safe_node_token(pathway_id)}__{safe_node_token(level)}"
    )


def require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{name} is missing columns {missing}; existing columns={list(frame.columns)}"
        )


def parse_ratio_text(value: str) -> Tuple[float, float, float]:
    pieces = [float(piece.strip()) for piece in value.split(",")]
    if len(pieces) != 3 or any(piece <= 0 for piece in pieces):
        raise argparse.ArgumentTypeError(
            "--split-ratios must contain three positive values, e.g. 0.70,0.15,0.15"
        )
    total = sum(pieces)
    return tuple(piece / total for piece in pieces)  # type: ignore[return-value]


def parse_donor_path_items(values: Optional[Sequence[str]], option: str) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"{option} must use donor=path syntax: {item!r}")
        donor_raw, path_raw = item.split("=", 1)
        donor = normalize_donor(donor_raw)
        if donor not in VALID_DONORS:
            raise ValueError(f"Unsupported donor in {option}: {donor_raw!r}")
        if donor in result:
            raise ValueError(f"Duplicate {option} entry for {donor}")
        result[donor] = Path(path_raw).expanduser().resolve()
    return result


def discover_h5_files(h5_dir: Path, overrides: Mapping[str, Path]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for donor in VALID_DONORS:
        result[donor] = overrides.get(
            donor,
            h5_dir / f"vdj_v1_hs_aggregated_{donor}_filtered_feature_bc_matrix.h5",
        )
    return result


def discover_vdj_files(vdj_dir: Path, overrides: Mapping[str, Path]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for donor in VALID_DONORS:
        result[donor] = overrides.get(
            donor,
            vdj_dir / f"vdj_v1_hs_aggregated_{donor}_all_contig_annotations.csv",
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the complete TCR/peptide/VDJ/four-state-UCell graph in one run."
    )
    binding = parser.add_mutually_exclusive_group(required=True)
    binding.add_argument(
        "--binding-file",
        type=Path,
        help=(
            "One CSV with donor,barcode,cdr3a,cdr3b,peptide,label. "
            "With --inference-only, label is optional and all rows are test data. "
            "Otherwise, if no split column exists, a deterministic TCR-sequence-disjoint split is created."
        ),
    )
    binding.add_argument(
        "--split-files",
        nargs=3,
        metavar=("TRAIN_CSV", "VAL_CSV", "TEST_CSV"),
        help=(
            "Three CSV files, each with the six required binding columns; with "
            "--inference-only, label is optional."
        ),
    )
    parser.add_argument(
        "--inference-only",
        action="store_true",
        help=(
            "Build an unlabeled external-inference graph. The input only needs "
            "donor, barcode, cdr3a, cdr3b and peptide; all rows are written to test_edges.csv."
        ),
    )
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument(
        "--h5-dir",
        type=Path,
        default=Path("inputs/10x/expression"),
        help="Directory for the donor expression H5 files (default: %(default)s).",
    )
    parser.add_argument(
        "--h5",
        action="append",
        help="Override one H5 path with donor=path; may be repeated.",
    )
    parser.add_argument(
        "--vdj-dir",
        type=Path,
        default=Path("inputs/10x/vdj"),
        help="Directory for the donor V(D)J annotation CSVs (default: %(default)s).",
    )
    parser.add_argument(
        "--vdj-contig",
        action="append",
        help="Override one all_contig_annotations.csv path with donor=path; may be repeated.",
    )
    parser.add_argument(
        "--skip-vdj",
        action="store_true",
        help="Build an intentionally incomplete graph without VDJ nodes/edges.",
    )
    parser.add_argument(
        "--state-annotations",
        type=Path,
        help=(
            "Existing selected_cell_4state_annotations.csv. If omitted, the four-state "
            "annotation is recomputed from the H5 matrices."
        ),
    )
    parser.add_argument(
        "--cluster-state-map",
        type=Path,
        help="Optional JSON mapping from Leiden cluster strings to one of the four states.",
    )
    parser.add_argument("--split-ratios", type=parse_ratio_text, default=(0.70, 0.15, 0.15))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-search-trials",
        type=int,
        default=256,
        help=(
            "For a single unsplit file, evaluate this many deterministic sequence-group "
            "assignments and retain the one closest to the requested row/positive ratios."
        ),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=6000)
    parser.add_argument("--max-mt-percent", type=float, default=20.0)
    parser.add_argument("--min-counts", type=int, default=500)
    parser.add_argument("--n-hvg", type=int, default=3000)
    parser.add_argument("--n-pcs", type=int, default=30)
    parser.add_argument("--n-neighbors", type=int, default=20)
    parser.add_argument("--leiden-resolution", type=float, default=0.30)
    parser.add_argument("--low-quantile", type=float, default=0.25)
    parser.add_argument("--high-quantile", type=float, default=0.75)
    parser.add_argument("--ucell-max-rank", type=int, default=1500)
    parser.add_argument("--ucell-chunk-size", type=int, default=500)
    parser.add_argument("--ucell-n-jobs", type=int, default=8)
    parser.add_argument("--min-pathway-genes", type=int, default=5)
    parser.add_argument("--max-unmatched-state-fraction", type=float, default=0.05)
    parser.add_argument(
        "--kegg-signatures-json",
        type=Path,
        help="Optional cached pathway signature JSON; avoids an online KEGG request.",
    )
    parser.add_argument("--refresh-kegg", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--write-combined", action="store_true")
    return parser


def ensure_output_directory(outdir: Path, overwrite: bool) -> Path:
    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    collisions = [outdir / name for name in GRAPH_OUTPUT_FILENAMES if (outdir / name).exists()]
    if collisions and not overwrite:
        raise FileExistsError(
            "Output graph files already exist. Use a new --outdir or pass --overwrite. "
            f"Examples: {[str(path) for path in collisions[:5]]}"
        )
    return outdir


def json_safe(value: object):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def load_one_binding_file(
    path: Path,
    split: Optional[str],
    require_label: bool = True,
) -> pd.DataFrame:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, low_memory=False)
    frame.columns = [clean_text(column) for column in frame.columns]
    required = (
        REQUIRED_BINDING_COLUMNS
        if require_label
        else REQUIRED_BINDING_COLUMNS[:-1]
    )
    require_columns(frame, required, str(path))
    keep = list(required)
    if not require_label and "label" in frame.columns:
        keep.append("label")
    if "split" in frame.columns:
        keep.append("split")
    frame = frame[keep].copy()
    frame["source_file"] = str(path)
    if split is not None:
        frame["split"] = split
    return frame


def clean_binding_table(
    frame: pd.DataFrame,
    inference_only: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    original = frame.copy()
    frame["donor"] = frame["donor"].map(normalize_donor)

    invalid_donor = ~frame["donor"].isin(VALID_DONORS)
    if invalid_donor.any():
        raise ValueError(
            "Binding table contains donors outside donor1-donor4: "
            f"{sorted(frame.loc[invalid_donor, 'donor'].unique().tolist())}"
        )

    cleaned_barcodes = []
    barcode_errors = []
    for index, value in frame["barcode"].items():
        try:
            cleaned_barcodes.append(canonical_barcode(value))
        except ValueError as exc:
            cleaned_barcodes.append("")
            barcode_errors.append({"row": int(index), "value": value, "error": str(exc)})
    if barcode_errors:
        raise ValueError(f"Invalid binding barcodes; examples={barcode_errors[:20]}")
    frame["barcode"] = cleaned_barcodes

    frame["cdr3a"] = [clean_sequence(value, "cdr3a") for value in frame["cdr3a"]]
    frame["cdr3b"] = [clean_sequence(value, "cdr3b") for value in frame["cdr3b"]]
    frame["peptide"] = frame["peptide"].map(clean_peptide)

    empty_mask = frame[["donor", "barcode", "cdr3a", "cdr3b", "peptide"]].eq("").any(axis=1)
    if empty_mask.any():
        examples = frame.loc[empty_mask, list(REQUIRED_BINDING_COLUMNS)].head(20)
        raise ValueError(
            "Binding rows contain empty donor/barcode/CDR3/peptide fields after cleaning:\n"
            + examples.to_string(index=False)
        )

    if inference_only:

        frame["label"] = np.nan
    else:
        numeric_label = pd.to_numeric(frame["label"], errors="coerce")
        invalid_label = numeric_label.isna() | ~numeric_label.isin([0, 1])
        if invalid_label.any():
            raise ValueError(
                "Binding labels must be 0 or 1; examples="
                + repr(frame.loc[invalid_label, "label"].head(20).tolist())
            )
        frame["label"] = numeric_label.astype(int)

    if "split" in frame.columns:
        frame["split"] = frame["split"].map(lambda value: clean_text(value).lower())

    duplicate_mask = frame.duplicated(
        list(REQUIRED_BINDING_COLUMNS) + (["split"] if "split" in frame.columns else []),
        keep="first",
    )
    duplicate_rows = original.loc[duplicate_mask].copy()
    frame = frame.loc[~duplicate_mask].reset_index(drop=True)
    return frame, duplicate_rows


def stable_sequence_split(
    cdr3a: str,
    cdr3b: str,
    ratios: Tuple[float, float, float],
    seed: int,
) -> str:
    digest = hashlib.md5(f"seed={seed}|cdr3a={cdr3a}|cdr3b={cdr3b}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / float(2**64)
    if value < ratios[0]:
        return "train"
    if value < ratios[0] + ratios[1]:
        return "val"
    return "test"


def assign_sequence_disjoint_splits(
    frame: pd.DataFrame,
    ratios: Tuple[float, float, float],
    seed: int,
    trials: int,
) -> Tuple[pd.Series, Dict[str, object]]:
    if trials < 1:
        raise ValueError("--split-search-trials must be at least 1.")
    group_stats = (
        frame.groupby(list(TCR_SEQUENCE_KEY), as_index=False)
        .agg(n_rows=("label", "size"), n_positive=("label", "sum"))
        .reset_index(drop=True)
    )
    split_names = np.array(["train", "val", "test"], dtype=object)
    target = np.asarray(ratios, dtype=float)
    total_rows = float(group_stats["n_rows"].sum())
    total_positive = float(group_stats["n_positive"].sum())
    total_groups = float(len(group_stats))
    best_objective = np.inf
    best_assignment: Optional[np.ndarray] = None
    best_trial_seed: Optional[int] = None

    keys = list(
        group_stats[["cdr3a", "cdr3b"]].itertuples(index=False, name=None)
    )
    row_weights = group_stats["n_rows"].to_numpy(dtype=float)
    positive_weights = group_stats["n_positive"].to_numpy(dtype=float)

    for trial in range(trials):
        trial_seed = seed + trial
        values = np.array(
            [
                int.from_bytes(
                    hashlib.md5(
                        f"seed={trial_seed}|cdr3a={cdr3a}|cdr3b={cdr3b}".encode("utf-8")
                    ).digest()[:8],
                    byteorder="big",
                    signed=False,
                )
                / float(2**64)
                for cdr3a, cdr3b in keys
            ],
            dtype=float,
        )
        assignment = np.where(
            values < target[0],
            0,
            np.where(values < target[0] + target[1], 1, 2),
        )
        group_counts = np.bincount(assignment, minlength=3).astype(float)
        if np.any(group_counts == 0):
            continue
        row_counts = np.bincount(assignment, weights=row_weights, minlength=3)
        positive_counts = np.bincount(assignment, weights=positive_weights, minlength=3)
        objective = float(np.abs(row_counts / total_rows - target).sum())
        objective += 0.25 * float(np.abs(group_counts / total_groups - target).sum())
        if total_positive > 0:
            objective += float(np.abs(positive_counts / total_positive - target).sum())
        if objective < best_objective:
            best_objective = objective
            best_assignment = assignment.copy()
            best_trial_seed = trial_seed

    if best_assignment is None or best_trial_seed is None:
        raise RuntimeError("Could not construct three nonempty TCR-sequence-disjoint splits.")

    mapping = {
        key: str(split_names[index])
        for key, index in zip(keys, best_assignment)
    }
    assigned = pd.Series(
        [mapping[(cdr3a, cdr3b)] for cdr3a, cdr3b in frame[["cdr3a", "cdr3b"]].itertuples(index=False, name=None)],
        index=frame.index,
        dtype="object",
    )
    details = {
        "candidate_trials": int(trials),
        "selected_trial_seed": int(best_trial_seed),
        "selection_objective": float(best_objective),
    }
    return assigned, details


def load_binding_inputs(args: argparse.Namespace) -> Tuple[pd.DataFrame, Dict[str, object]]:
    inference_only = bool(getattr(args, "inference_only", False))
    if args.binding_file is not None:
        raw = load_one_binding_file(
            args.binding_file,
            split=None,
            require_label=not inference_only,
        )
        input_mode = "single_file"
    else:
        paths = [Path(item) for item in args.split_files]
        raw = pd.concat(
            [
                load_one_binding_file(
                    paths[0], "train", require_label=not inference_only
                ),
                load_one_binding_file(
                    paths[1], "val", require_label=not inference_only
                ),
                load_one_binding_file(
                    paths[2], "test", require_label=not inference_only
                ),
            ],
            ignore_index=True,
        )
        input_mode = "three_split_files"

    cleaned, duplicate_rows = clean_binding_table(
        raw,
        inference_only=inference_only,
    )

    split_search_details: Dict[str, object] = {}
    if inference_only:
        if cleaned.empty:
            raise ValueError("The inference-only binding table is empty.")
        cleaned["split"] = "test"
        split_method = "inference_only_all_rows_test"
    elif input_mode == "single_file":
        if "split" in cleaned.columns:
            observed_split_values = set(cleaned["split"])
            if not observed_split_values <= {"train", "val", "test"}:
                raise ValueError(
                    f"The input split column contains unsupported values: {sorted(observed_split_values)}"
                )
            if cleaned["split"].eq("").any():
                raise ValueError("The split column contains empty values.")
            split_method = "input_split_column"
        else:
            cleaned["split"], split_search_details = assign_sequence_disjoint_splits(
                cleaned,
                args.split_ratios,
                args.split_seed,
                args.split_search_trials,
            )
            split_method = "optimized_deterministic_hash_of_paired_cdr3_sequence"
    else:
        split_method = "input_files"

    if not inference_only:
        invalid_splits = sorted(set(cleaned["split"]) - {"train", "val", "test"})
        if invalid_splits:
            raise ValueError(f"Unsupported split labels: {invalid_splits}")
        for split in ("train", "val", "test"):
            if not cleaned["split"].eq(split).any():
                raise ValueError(f"Split {split!r} is empty.")

    conflict = (
        cleaned.groupby(list(TCR_INSTANCE_KEY) + ["peptide"])["label"]
        .nunique()
        .reset_index(name="n_labels")
    )
    if conflict["n_labels"].gt(1).any():
        raise ValueError(
            "The same TCR-instance/peptide pair has conflicting labels; examples="
            + conflict.loc[conflict["n_labels"].gt(1)].head(20).to_json(orient="records")
        )

    pair_split = (
        cleaned.groupby(list(TCR_INSTANCE_KEY) + ["peptide"])["split"]
        .nunique()
        .reset_index(name="n_splits")
    )
    if pair_split["n_splits"].gt(1).any():
        raise ValueError("The same TCR-instance/peptide pair occurs in multiple splits.")

    sequence_split = (
        cleaned.groupby(list(TCR_SEQUENCE_KEY))["split"]
        .nunique()
        .reset_index(name="n_splits")
    )
    if sequence_split["n_splits"].gt(1).any():
        bad = sequence_split.loc[sequence_split["n_splits"].gt(1)].head(20)
        raise ValueError(
            "Paired TCR sequences occur in multiple splits; examples="
            + bad.to_json(orient="records")
        )

    summary = {
        "input_mode": input_mode,
        "inference_only": inference_only,
        "split_method": split_method,
        "n_rows_after_cleaning": int(len(cleaned)),
        "n_exact_duplicate_rows_removed": int(len(duplicate_rows)),
        "split_rows": cleaned["split"].value_counts().sort_index().astype(int).to_dict(),
        "split_positive_rows": (
            cleaned.loc[cleaned["label"].eq(1), "split"].value_counts().sort_index().astype(int).to_dict()
        ),
        "split_search": split_search_details,
    }
    return cleaned, summary


def build_base_graph(binding: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    instances = binding[list(TCR_INSTANCE_KEY)].drop_duplicates().copy()
    instances["node_id"] = [
        make_tcr_instance_id(donor, barcode, cdr3a, cdr3b)
        for donor, barcode, cdr3a, cdr3b in instances.itertuples(index=False, name=None)
    ]
    instances["tcr_sequence_id"] = [
        make_tcr_sequence_id(cdr3a, cdr3b)
        for cdr3a, cdr3b in instances[["cdr3a", "cdr3b"]].itertuples(index=False, name=None)
    ]
    instances.insert(1, "node_type", "TCR_instance")
    instances = instances[
        ["node_id", "node_type", "donor", "barcode", "cdr3a", "cdr3b", "tcr_sequence_id"]
    ].sort_values("node_id").reset_index(drop=True)

    if instances["node_id"].duplicated().any():
        raise RuntimeError("TCR instance hash collision or duplicated node_id detected.")

    sequence_nodes = (
        instances.groupby("tcr_sequence_id", as_index=False)
        .agg(
            cdr3a=("cdr3a", "first"),
            cdr3b=("cdr3b", "first"),
            n_instances=("node_id", "nunique"),
            n_donors=("donor", "nunique"),
            n_barcodes=("barcode", "nunique"),
        )
        .rename(columns={"tcr_sequence_id": "node_id"})
        .sort_values("node_id")
        .reset_index(drop=True)
    )
    sequence_nodes.insert(1, "node_type", "TCR_sequence")

    sequence_instance_edges = (
        instances[["tcr_sequence_id", "node_id"]]
        .rename(columns={"tcr_sequence_id": "src", "node_id": "dst"})
        .drop_duplicates()
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
    )
    sequence_instance_edges["edge_type"] = "tcr_sequence_to_instance"
    sequence_instance_edges = sequence_instance_edges[["src", "dst", "edge_type"]]

    peptides = binding[["peptide"]].drop_duplicates().sort_values("peptide").reset_index(drop=True)
    peptides.insert(0, "node_id", peptides["peptide"].map(make_peptide_id))
    peptides.insert(1, "node_type", "peptide")
    peptides["donor"] = ""
    peptides["barcode"] = ""
    peptides = peptides[["node_id", "node_type", "donor", "barcode", "peptide"]]

    instance_lookup = instances[
        ["node_id", "donor", "barcode", "cdr3a", "cdr3b", "tcr_sequence_id"]
    ].rename(columns={"node_id": "src"})
    peptide_lookup = peptides[["node_id", "peptide"]].rename(columns={"node_id": "dst"})

    binding_edges: Dict[str, pd.DataFrame] = {}
    for split in ("train", "val", "test"):
        block = binding.loc[binding["split"].eq(split), list(REQUIRED_BINDING_COLUMNS)].copy()
        block = block.merge(
            instance_lookup,
            on=["donor", "barcode", "cdr3a", "cdr3b"],
            how="left",
            validate="many_to_one",
        )
        block = block.merge(peptide_lookup, on="peptide", how="left", validate="many_to_one")
        if block[["src", "dst", "tcr_sequence_id"]].isna().any().any():
            raise RuntimeError(f"{split} binding rows failed to map to graph nodes.")
        block["edge_type"] = "tcr_instance_peptide_binding"
        block = block[
            [
                "src",
                "dst",
                "edge_type",
                "label",
                "donor",
                "barcode",
                "cdr3a",
                "cdr3b",
                "peptide",
                "tcr_sequence_id",
            ]
        ].drop_duplicates()
        binding_edges[split] = block.sort_values(["src", "dst"]).reset_index(drop=True)

    return {
        "tcr_instance_nodes": instances,
        "tcr_sequence_nodes": sequence_nodes,
        "peptide_nodes": peptides,
        "tcr_sequence_to_instance_edges": sequence_instance_edges,
        "train_edges": binding_edges["train"],
        "val_edges": binding_edges["val"],
        "test_edges": binding_edges["test"],
    }




def pick_column(frame: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lookup = {str(column).strip().lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "t"})


def clean_gene(value: object) -> str:
    gene = clean_text(value)
    return gene.upper() if gene else ""


def build_exact_core_index(values: Iterable[str]) -> Tuple[set, Dict[str, List[str]]]:
    exact = {clean_text(value).upper() for value in values if clean_text(value)}
    core_to_values: Dict[str, List[str]] = {}
    for value in sorted(exact):
        core_to_values.setdefault(barcode_core(value), []).append(value)
    return exact, core_to_values


def match_barcode_exact_then_unique_core(
    requested: str,
    exact_values: set,
    core_to_values: Mapping[str, Sequence[str]],
) -> Tuple[str, str]:
    requested = clean_text(requested).upper()
    if requested in exact_values:
        return requested, "exact"
    candidates = list(core_to_values.get(barcode_core(requested), []))
    if len(candidates) == 1:
        return candidates[0], "core_unique"
    if len(candidates) > 1:
        return "", "ambiguous_core"
    return "", "unmatched"


def empty_vdj_outputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nodes = pd.DataFrame(columns=["node_id", "node_type", "vdj_type", "gene"])
    edges = pd.DataFrame(columns=["src", "dst", "edge_type"])
    audit = pd.DataFrame(
        columns=[
            "donor",
            "contig_row",
            "contig_barcode",
            "matched_instance_barcode",
            "match_method",
            "chain",
            "vdj_type",
            "gene",
        ]
    )
    unmatched = pd.DataFrame(
        columns=["node_id", "donor", "barcode", "cdr3a", "cdr3b", "reason"]
    )
    return nodes, edges, audit, unmatched


def build_vdj_graph(
    instances: pd.DataFrame,
    vdj_files: Mapping[str, Path],
    skip_vdj: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if skip_vdj:
        return empty_vdj_outputs()

    target_by_donor = {
        donor: sorted(
            instances.loc[instances["donor"].eq(donor), "barcode"].drop_duplicates().tolist()
        )
        for donor in VALID_DONORS
    }

    record_rows: List[Dict[str, object]] = []
    audit_rows: List[Dict[str, object]] = []

    for donor in VALID_DONORS:
        if not target_by_donor[donor]:
            continue
        path = Path(vdj_files[donor])
        if not path.exists():
            raise FileNotFoundError(
                f"Missing VDJ contig table for {donor}: {path}. "
                "The expression H5 cannot replace all_contig_annotations.csv."
            )

        contigs = pd.read_csv(path, low_memory=False)
        contigs.columns = [clean_text(column) for column in contigs.columns]
        barcode_column = pick_column(contigs, ("barcode",))
        chain_column = pick_column(contigs, ("chain",))
        v_column = pick_column(contigs, ("v_gene", "v_call", "v"))
        j_column = pick_column(contigs, ("j_gene", "j_call", "j"))
        if None in (barcode_column, chain_column, v_column, j_column):
            raise ValueError(
                f"{path} must contain barcode, chain, v_gene and j_gene columns; "
                f"existing columns={list(contigs.columns)}"
            )

        if "is_cell" in contigs.columns:
            contigs = contigs.loc[parse_bool_series(contigs["is_cell"])].copy()
        if "high_confidence" in contigs.columns:
            contigs = contigs.loc[parse_bool_series(contigs["high_confidence"])].copy()
        if "productive" in contigs.columns:
            contigs = contigs.loc[parse_bool_series(contigs["productive"])].copy()

        exact_targets, core_targets = build_exact_core_index(target_by_donor[donor])

        for row_index, row in contigs.iterrows():
            chain = clean_text(row[chain_column]).upper()  # type: ignore[index]
            if chain not in {"TRA", "TRB"}:
                continue
            try:
                contig_barcode = canonical_barcode(row[barcode_column], "VDJ barcode")  # type: ignore[index]
            except ValueError:
                audit_rows.append(
                    {
                        "donor": donor,
                        "contig_row": int(row_index),
                        "contig_barcode": clean_text(row[barcode_column]),  # type: ignore[index]
                        "matched_instance_barcode": "",
                        "match_method": "invalid_barcode",
                        "chain": chain,
                        "vdj_type": "",
                        "gene": "",
                    }
                )
                continue

            matched_barcode, match_method = match_barcode_exact_then_unique_core(
                contig_barcode,
                exact_targets,
                core_targets,
            )

            gene_pairs = (
                (("TRAV", clean_gene(row[v_column])), ("TRAJ", clean_gene(row[j_column])))  # type: ignore[index]
                if chain == "TRA"
                else (("TRBV", clean_gene(row[v_column])), ("TRBJ", clean_gene(row[j_column])))  # type: ignore[index]
            )

            for vdj_type, gene in gene_pairs:
                audit_rows.append(
                    {
                        "donor": donor,
                        "contig_row": int(row_index),
                        "contig_barcode": contig_barcode,
                        "matched_instance_barcode": matched_barcode,
                        "match_method": match_method,
                        "chain": chain,
                        "vdj_type": vdj_type,
                        "gene": gene,
                    }
                )
                if matched_barcode and gene:
                    record_rows.append(
                        {
                            "donor": donor,
                            "barcode": matched_barcode,
                            "vdj_type": vdj_type,
                            "gene": gene,
                            "vdj_node_id": f"{vdj_type}|{gene}",
                        }
                    )

    if not record_rows:
        raise RuntimeError("No VDJ records matched the TCR instance barcodes.")

    records = pd.DataFrame(record_rows).drop_duplicates().reset_index(drop=True)
    nodes = (
        records[["vdj_node_id", "vdj_type", "gene"]]
        .drop_duplicates()
        .rename(columns={"vdj_node_id": "node_id"})
        .sort_values(["vdj_type", "gene"])
        .reset_index(drop=True)
    )
    nodes.insert(1, "node_type", "vdj_gene")
    nodes = nodes[["node_id", "node_type", "vdj_type", "gene"]]

    instance_gene = instances[
        ["node_id", "tcr_sequence_id", "donor", "barcode"]
    ].merge(
        records[["donor", "barcode", "vdj_node_id"]],
        on=["donor", "barcode"],
        how="inner",
        validate="many_to_many",
    )
    edges = (
        instance_gene[["tcr_sequence_id", "vdj_node_id"]]
        .drop_duplicates()
        .rename(columns={"tcr_sequence_id": "src", "vdj_node_id": "dst"})
        .sort_values(["src", "dst"])
        .reset_index(drop=True)
    )
    edges["edge_type"] = "tcr_sequence_to_vdj_gene"
    edges = edges[["src", "dst", "edge_type"]]

    matched_instance_ids = set(instance_gene["node_id"])
    unmatched = instances.loc[
        ~instances["node_id"].isin(matched_instance_ids),
        ["node_id", "donor", "barcode", "cdr3a", "cdr3b"],
    ].copy()
    unmatched["reason"] = "no_matched_productive_TRA_TRB_vdj_gene"
    audit = pd.DataFrame(audit_rows)
    return nodes, edges, audit, unmatched.reset_index(drop=True)



def import_single_cell_stack():
    try:
        import anndata as ad
        import harmonypy as hm
        import scanpy as sc
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError(
            "Four-state annotation requires scanpy, anndata, harmonypy and scipy."
        ) from exc
    return ad, hm, sc, sp


def load_cluster_state_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        mapping = dict(DEFAULT_CLUSTER_STATE_MAP)
    else:
        with open(path.expanduser().resolve(), "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict):
            raise ValueError("--cluster-state-map must contain one JSON object.")
        mapping = {str(key): clean_text(value) for key, value in raw.items()}
    unsupported = sorted(set(mapping.values()) - set(FOUR_STATES))
    if unsupported:
        raise ValueError(f"Cluster-state mapping contains unsupported states: {unsupported}")
    return mapping


def build_requested_cell_table(binding: pd.DataFrame) -> pd.DataFrame:
    requested = (
        binding.groupby(["donor", "barcode"], as_index=False)
        .agg(split_list=("split", lambda values: ";".join(sorted(set(map(str, values))))))
        .sort_values(["donor", "barcode"])
        .reset_index(drop=True)
    )
    return requested


def map_requested_barcodes_to_available(
    requested: Sequence[str],
    available: Sequence[str],
) -> Tuple[Dict[str, str], Dict[str, str], List[str]]:
    exact_available, core_available = build_exact_core_index(available)
    mapping: Dict[str, str] = {}
    methods: Dict[str, str] = {}
    unmatched: List[str] = []
    for barcode in requested:
        matched, method = match_barcode_exact_then_unique_core(
            barcode,
            exact_available,
            core_available,
        )
        if matched:
            mapping[barcode] = matched
            methods[barcode] = method
        else:
            unmatched.append(barcode)
    if len(set(mapping.values())) != len(mapping):
        reverse: Dict[str, List[str]] = defaultdict(list)
        for requested_barcode, matched_barcode in mapping.items():
            reverse[matched_barcode].append(requested_barcode)
        collisions = {key: values for key, values in reverse.items() if len(values) > 1}
        raise ValueError(
            "Multiple requested barcodes map to the same H5 cell; examples="
            + repr(dict(list(collisions.items())[:20]))
        )
    return mapping, methods, unmatched


def run_harmony_compatible(
    adata_hvg,
    hm,
    random_seed: int,
    source_basis: str = "X_pca",
    target_basis: str = "X_pca_harmony",
) -> str:
    source = np.asarray(adata_hvg.obsm[source_basis], dtype=np.float64)
    result = hm.run_harmony(
        data_mat=source,
        meta_data=adata_hvg.obs.copy(),
        vars_use=["donor"],
        random_state=random_seed,
    )
    corrected = np.asarray(result.Z_corr, dtype=np.float32)
    if corrected.shape == source.shape:
        final = corrected
    elif corrected.T.shape == source.shape:
        final = corrected.T
    else:
        raise ValueError(
            f"Unexpected Harmony shape: PCA={source.shape}, Z_corr={corrected.shape}"
        )
    if not np.isfinite(final).all():
        raise ValueError("Harmony output contains NaN or infinity.")
    adata_hvg.obsm[target_basis] = final
    return target_basis


def compute_four_state_annotations(
    binding: pd.DataFrame,
    h5_files: Mapping[str, Path],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    ad, hm, sc, _sp = import_single_cell_stack()
    np.random.seed(args.random_seed)

    requested = build_requested_cell_table(binding)
    selected_adatas = []
    matching_summary: List[Dict[str, object]] = []

    for donor in VALID_DONORS:
        donor_requested = requested.loc[requested["donor"].eq(donor)].copy()
        if donor_requested.empty:
            continue
        h5_path = Path(h5_files[donor])
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        donor_adata = sc.read_10x_h5(str(h5_path), gex_only=True)
        donor_adata.var_names_make_unique()
        donor_adata.obs_names = pd.Index(
            [clean_text(value).upper() for value in donor_adata.obs_names],
            dtype="object",
        )
        if donor_adata.obs_names.duplicated().any():
            raise ValueError(f"{h5_path} contains duplicated cell barcodes.")

        requested_barcodes = donor_requested["barcode"].tolist()
        mapping, methods, unmatched = map_requested_barcodes_to_available(
            requested_barcodes,
            donor_adata.obs_names.astype(str).tolist(),
        )
        if unmatched:
            print(
                f"[WARN] {donor}: {len(unmatched)} selected barcodes are absent or "
                "ambiguous in H5 and will be skipped during four-state annotation; "
                f"examples={unmatched[:20]}"
            )

        selected_requested = [barcode for barcode in requested_barcodes if barcode in mapping]
        if not selected_requested:
            raise RuntimeError(f"{donor}: no selected barcode can be matched to H5")
        selected_h5 = [mapping[barcode] for barcode in selected_requested]
        donor_selected = donor_adata[selected_h5, :].copy()
        metadata = donor_requested.set_index("barcode").loc[selected_requested]
        donor_selected.obs["donor"] = donor
        donor_selected.obs["input_barcode"] = selected_requested
        donor_selected.obs["barcode_core"] = selected_requested
        donor_selected.obs["split_list"] = metadata["split_list"].to_numpy()
        donor_selected.obs["original_obs_name"] = selected_h5
        donor_selected.obs_names = pd.Index(
            [f"{donor}__{barcode}" for barcode in selected_h5],
            dtype="object",
        )
        selected_adatas.append(donor_selected)
        matching_summary.append(
            {
                "donor": donor,
                "n_h5_cells": int(donor_adata.n_obs),
                "n_requested": int(len(requested_barcodes)),
                "n_exact": int(sum(method == "exact" for method in methods.values())),
                "n_core_unique": int(sum(method == "core_unique" for method in methods.values())),
                "n_unmatched": int(len(unmatched)),
            }
        )

    if not selected_adatas:
        raise RuntimeError("No binding-data cells were selected from the H5 matrices.")

    adata = ad.concat(selected_adatas, join="inner", merge="same", index_unique=None)
    adata.var_names_make_unique()
    adata.var["mt"] = adata.var_names.astype(str).str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )
    before_qc = int(adata.n_obs)
    keep = (
        adata.obs["n_genes_by_counts"].ge(args.min_genes)
        & adata.obs["n_genes_by_counts"].le(args.max_genes)
        & adata.obs["pct_counts_mt"].le(args.max_mt_percent)
        & adata.obs["total_counts"].ge(args.min_counts)
    )
    adata = adata[keep].copy()
    if adata.n_obs < 10:
        raise RuntimeError(f"Only {adata.n_obs} cells remain after QC.")

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata.copy()

    if importlib.util.find_spec("skmisc") is not None:
        hvg_method = "seurat_v3"
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=args.n_hvg,
            flavor="seurat_v3",
            layer="counts",
            batch_key="donor",
            subset=False,
        )
    else:
        hvg_method = "seurat"
        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=args.n_hvg,
            flavor="seurat",
            batch_key="donor",
            subset=False,
        )
    n_hvg = int(adata.var["highly_variable"].sum())
    if n_hvg < 50:
        raise RuntimeError(f"Too few highly variable genes: {n_hvg}")

    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    resolved_n_pcs = min(args.n_pcs, adata_hvg.n_obs - 1, adata_hvg.n_vars - 1)
    if resolved_n_pcs < 2:
        raise RuntimeError("Insufficient cells or genes for PCA.")
    sc.tl.pca(
        adata_hvg,
        n_comps=resolved_n_pcs,
        svd_solver="arpack",
        random_state=args.random_seed,
    )
    harmony_basis = run_harmony_compatible(adata_hvg, hm, args.random_seed)
    sc.pp.neighbors(
        adata_hvg,
        n_neighbors=args.n_neighbors,
        n_pcs=resolved_n_pcs,
        use_rep=harmony_basis,
        random_state=args.random_seed,
    )
    sc.tl.leiden(
        adata_hvg,
        resolution=args.leiden_resolution,
        random_state=args.random_seed,
        key_added="leiden",
    )

    cluster_map = load_cluster_state_map(args.cluster_state_map)
    observed_clusters = sorted(set(adata_hvg.obs["leiden"].astype(str)))
    missing_clusters = sorted(set(observed_clusters) - set(cluster_map))
    if missing_clusters:
        raise RuntimeError(
            "Leiden produced clusters without a reviewed four-state label: "
            f"{missing_clusters}. Supply --cluster-state-map after marker review."
        )
    adata.obs["leiden"] = adata_hvg.obs.loc[adata.obs_names, "leiden"].astype(str).to_numpy()
    adata.obs["Tcellstate_manual"] = adata.obs["leiden"].map(cluster_map)
    if adata.obs["Tcellstate_manual"].isna().any():
        raise RuntimeError("Some cells did not receive a four-state annotation.")

    annotations = pd.DataFrame(
        {
            "adata_obs_name": adata.obs_names.astype(str),
            "donor": adata.obs["donor"].astype(str).to_numpy(),
            "original_obs_name": adata.obs["original_obs_name"].astype(str).to_numpy(),
            "input_barcode": adata.obs["input_barcode"].astype(str).to_numpy(),
            "barcode_core": adata.obs["barcode_core"].astype(str).to_numpy(),
            "split_list": adata.obs["split_list"].astype(str).to_numpy(),
            "leiden": adata.obs["leiden"].astype(str).to_numpy(),
            "Tcellstate_manual": adata.obs["Tcellstate_manual"].astype(str).to_numpy(),
        }
    ).drop_duplicates().reset_index(drop=True)
    unsupported = sorted(set(annotations["Tcellstate_manual"]) - set(FOUR_STATES))
    if unsupported:
        raise RuntimeError(f"Unsupported state labels in annotation output: {unsupported}")

    summary = {
        "mode": "computed_from_h5",
        "n_selected_before_qc": before_qc,
        "n_selected_after_qc": int(len(annotations)),
        "n_hvg": n_hvg,
        "hvg_method": hvg_method,
        "n_pcs": int(resolved_n_pcs),
        "leiden_resolution": float(args.leiden_resolution),
        "cluster_state_map": cluster_map,
        "cluster_counts": annotations["leiden"].value_counts().sort_index().astype(int).to_dict(),
        "state_counts": annotations["Tcellstate_manual"].value_counts().sort_index().astype(int).to_dict(),
        "barcode_matching": matching_summary,
    }
    return annotations, summary


def load_existing_state_annotations(path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    annotations = pd.read_csv(path, dtype=str, low_memory=False)
    require_columns(annotations, ("donor", "Tcellstate_manual"), str(path))
    if not any(
        column in annotations.columns
        for column in ("adata_obs_name", "original_obs_name", "input_barcode", "barcode_core", "barcode")
    ):
        raise ValueError(f"{path} has no usable barcode column.")
    annotations["donor"] = annotations["donor"].map(normalize_donor)
    annotations["Tcellstate_manual"] = annotations["Tcellstate_manual"].map(clean_text)
    unsupported = sorted(set(annotations["Tcellstate_manual"]) - set(FOUR_STATES))
    if unsupported:
        raise ValueError(f"Existing annotations contain unsupported states: {unsupported}")
    annotations = annotations.loc[annotations["donor"].isin(VALID_DONORS)].copy()
    return annotations.reset_index(drop=True), {
        "mode": "existing_annotation_csv",
        "source": str(path),
        "n_rows": int(len(annotations)),
        "state_counts": annotations["Tcellstate_manual"].value_counts().sort_index().astype(int).to_dict(),
    }


# ---------------------------------------------------------------------------
# Match TCR instances to the four-state annotation
# ---------------------------------------------------------------------------

def choose_annotation_barcode(row: pd.Series) -> str:
    for column in (
        "original_obs_name",
        "adata_obs_name",
        "input_barcode",
        "barcode",
        "barcode_core",
    ):
        if column not in row.index:
            continue
        candidates = extract_exact_10x_barcodes(row[column])
        suffixed = [value for value in candidates if re.search(r"-\d+$", value)]
        if len(suffixed) == 1:
            return suffixed[0]
        if len(candidates) == 1:
            return candidates[0]
    raise ValueError(
        "Cannot determine one canonical barcode for annotation row: "
        + repr(row.to_dict())
    )


def build_annotation_lookup(annotations: pd.DataFrame):
    exact_states: Dict[Tuple[str, str], set] = defaultdict(set)
    exact_rows: Dict[Tuple[str, str], set] = defaultdict(set)
    core_barcodes: Dict[Tuple[str, str], set] = defaultdict(set)
    core_states: Dict[Tuple[str, str], set] = defaultdict(set)
    core_rows: Dict[Tuple[str, str], set] = defaultdict(set)

    for row_index, row in annotations.iterrows():
        donor = normalize_donor(row["donor"])
        state = clean_text(row["Tcellstate_manual"])
        full_barcode = choose_annotation_barcode(row)
        exact_key = (donor, full_barcode)
        core_key = (donor, barcode_core(full_barcode))
        exact_states[exact_key].add(state)
        exact_rows[exact_key].add(int(row_index))
        core_barcodes[core_key].add(full_barcode)
        core_states[core_key].add(state)
        core_rows[core_key].add(int(row_index))

    exact_conflicts = [
        {"donor": key[0], "barcode": key[1], "states": sorted(states)}
        for key, states in exact_states.items()
        if len(states) > 1
    ]
    if exact_conflicts:
        raise ValueError(
            "One donor/full-barcode maps to multiple states; examples="
            + repr(exact_conflicts[:20])
        )

    unique_core_states: Dict[Tuple[str, str], set] = {}
    unique_core_rows: Dict[Tuple[str, str], set] = {}
    ambiguous_cores = set()
    for key, barcodes in core_barcodes.items():
        if len(barcodes) == 1:
            unique_core_states[key] = core_states[key]
            unique_core_rows[key] = core_rows[key]
        else:
            ambiguous_cores.add(key)
    return exact_states, exact_rows, unique_core_states, unique_core_rows, ambiguous_cores


def match_instances_to_states(
    instances: pd.DataFrame,
    annotations: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    (
        exact_states,
        exact_rows,
        unique_core_states,
        unique_core_rows,
        ambiguous_cores,
    ) = build_annotation_lookup(annotations)

    rows: List[Dict[str, object]] = []
    for instance in instances.itertuples(index=False):
        exact_key = (instance.donor, instance.barcode.upper())
        states = set(exact_states.get(exact_key, set()))
        annotation_rows = set(exact_rows.get(exact_key, set()))
        matched_keys = {instance.barcode.upper()} if states else set()
        method = "exact" if states else ""
        status = ""

        if not states:
            core_key = (instance.donor, barcode_core(instance.barcode))
            if core_key in ambiguous_cores:
                status = "ambiguous_barcode_core"
            else:
                states = set(unique_core_states.get(core_key, set()))
                annotation_rows = set(unique_core_rows.get(core_key, set()))
                if states:
                    matched_keys.add(core_key[1])
                    method = "core_unique"

        if len(states) > 1:
            raise ValueError(
                f"TCR instance {instance.node_id} matched multiple states: {sorted(states)}"
            )
        if len(states) == 1:
            state = next(iter(states))
            status = "matched"
        else:
            state = ""
            status = status or "unmatched_annotation"

        rows.append(
            {
                "node_id": instance.node_id,
                "donor": instance.donor,
                "barcode": instance.barcode.upper(),
                "match_status": status,
                "match_method": method,
                "Tcellstate": state,
                "matched_barcode_keys": ";".join(sorted(matched_keys)),
                "matched_annotation_rows": ";".join(map(str, sorted(annotation_rows))),
            }
        )

    audit = pd.DataFrame(rows)
    matched = audit.loc[audit["match_status"].eq("matched")].copy().reset_index(drop=True)
    return audit, matched


# ---------------------------------------------------------------------------
# KEGG pathway signatures and UCell scores
# ---------------------------------------------------------------------------

def pathway_catalog_frame() -> pd.DataFrame:
    return pd.DataFrame(PATHWAY_CATALOG, columns=["pathway_id", "pathway_name"])


def import_ucell_stack():
    try:
        import anndata as ad
        import h5py
        import pyucell as uc
        import scanpy as sc
        import scipy.sparse as sp
    except ImportError as exc:
        raise ImportError(
            "UCell construction requires scanpy, anndata, h5py, scipy and pyucell."
        ) from exc
    return ad, h5py, uc, sc, sp


def make_kegg_session():
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
    except ImportError as exc:
        raise ImportError("Online KEGG retrieval requires requests and urllib3.") from exc
    retry = Retry(
        total=6,
        connect=6,
        read=6,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": "complete-TCR-graph-builder/1.0"})
    return session


def request_kegg_text(session, endpoint: str, timeout: int = 120) -> str:
    url = endpoint if endpoint.startswith("http") else f"https://rest.kegg.jp/{endpoint.lstrip('/')}"
    response = session.get(url, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"KEGG request failed: status={response.status_code}, endpoint={endpoint}, "
            f"response={response.text[:500]!r}"
        )
    text = response.text.strip()
    if not text:
        raise RuntimeError(f"KEGG returned empty content for {endpoint}")
    time.sleep(0.10)
    return text


def chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def fetch_pathway_gene_ids(session, pathway_id: str) -> List[str]:
    text = request_kegg_text(session, f"link/hsa/{pathway_id}")
    result = {
        field.strip()
        for line in text.splitlines()
        for field in line.split("\t")
        if re.fullmatch(r"hsa:\d+", field.strip())
    }
    if not result:
        raise RuntimeError(f"No human genes were returned for {pathway_id}")
    return sorted(result, key=lambda value: int(value.split(":", 1)[1]))


def parse_kegg_primary_symbol(value: str) -> str:
    symbol = value.split(";", 1)[0].split(",", 1)[0].strip().upper()
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9._-]*", symbol):
        return ""
    if symbol in {"CDS", "RNA", "TRNA", "RRNA", "NCRNA", "PSEUDOGENE"}:
        return ""
    return symbol


def fetch_gene_symbols(session, gene_ids: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for batch in chunks(list(gene_ids), 10):
        text = request_kegg_text(session, "list/" + "+".join(batch))
        for line in text.splitlines():
            fields = [field.strip() for field in line.split("\t")]
            if len(fields) < 2 or not re.fullmatch(r"hsa:\d+", fields[0]):
                continue
            symbol = parse_kegg_primary_symbol(fields[-1])
            if symbol:
                mapping[fields[0]] = symbol
    return mapping


def fetch_kegg_signatures() -> Dict[str, object]:
    session = make_kegg_session()
    release_info = request_kegg_text(session, "info/pathway")
    pathway_gene_ids = {
        pathway_id: fetch_pathway_gene_ids(session, pathway_id)
        for pathway_id, _pathway_name in PATHWAY_CATALOG
    }
    all_ids = sorted(
        set().union(*(set(values) for values in pathway_gene_ids.values())),
        key=lambda value: int(value.split(":", 1)[1]),
    )
    symbol_map = fetch_gene_symbols(session, all_ids)
    signatures = {
        pathway_id: sorted(
            {symbol_map[gene_id] for gene_id in gene_ids if gene_id in symbol_map}
        )
        for pathway_id, gene_ids in pathway_gene_ids.items()
    }
    missing = sorted(set(all_ids) - set(symbol_map), key=lambda value: int(value.split(":", 1)[1]))
    for pathway_id, genes in signatures.items():
        if not genes:
            raise RuntimeError(f"No gene symbols were resolved for {pathway_id}")
    metadata = [
        {
            "pathway_id": pathway_id,
            "n_kegg_gene_ids": len(pathway_gene_ids[pathway_id]),
            "n_gene_symbols": len(signatures[pathway_id]),
            "n_gene_ids_without_symbol": len(
                set(pathway_gene_ids[pathway_id]) - set(symbol_map)
            ),
        }
        for pathway_id, _pathway_name in PATHWAY_CATALOG
    ]
    return {
        "source": "online_KEGG_REST",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "release_info": release_info,
        "signatures": signatures,
        "metadata": metadata,
        "gene_ids_without_symbol": missing,
    }


def load_or_fetch_kegg_signatures(
    args: argparse.Namespace,
    outdir: Path,
) -> Tuple[Dict[str, List[str]], Dict[str, object], Path]:
    if args.kegg_signatures_json is not None:
        cache_path = args.kegg_signatures_json.expanduser().resolve()
    else:
        cache_path = outdir / "kegg_pathway_signatures.json"

    if cache_path.exists() and not args.refresh_kegg:
        with open(cache_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        if args.kegg_signatures_json is not None and not cache_path.exists():
            raise FileNotFoundError(cache_path)
        payload = fetch_kegg_signatures()
        with open(cache_path, "w", encoding="utf-8") as handle:
            json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)

    if isinstance(payload, dict) and "signatures" in payload:
        raw_signatures = payload["signatures"]
    else:
        raw_signatures = payload
        payload = {"source": "user_JSON", "signatures": raw_signatures}
    if not isinstance(raw_signatures, dict):
        raise ValueError(f"Invalid KEGG signature JSON: {cache_path}")
    signatures = {
        clean_text(pathway_id).lower(): sorted(
            {clean_text(gene).upper() for gene in genes if clean_text(gene)}
        )
        for pathway_id, genes in raw_signatures.items()
    }
    missing_pathways = sorted(set(dict(PATHWAY_CATALOG)) - set(signatures))
    if missing_pathways:
        raise ValueError(f"KEGG signature JSON is missing pathways: {missing_pathways}")
    return signatures, payload, cache_path


def decode_h5_strings(values) -> List[str]:
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]


def read_10x_h5_gene_expression(h5_path: Path):
    ad, h5py, _uc, sc, _sp = import_ucell_stack()
    adata_full = sc.read_10x_h5(str(h5_path), gex_only=False)
    with h5py.File(h5_path, "r") as handle:
        if "matrix" not in handle or "features" not in handle["matrix"]:
            raise ValueError(f"{h5_path} lacks /matrix/features")
        features = handle["matrix"]["features"]
        if "name" not in features:
            raise ValueError(f"{h5_path} lacks /matrix/features/name")
        gene_names = decode_h5_strings(features["name"][:])
        feature_key = next(
            (key for key in ("feature_type", "feature_types") if key in features),
            None,
        )
        feature_types = (
            decode_h5_strings(features[feature_key][:])
            if feature_key is not None
            else ["Gene Expression"] * len(gene_names)
        )
    if len(gene_names) != adata_full.n_vars:
        raise ValueError(
            f"{h5_path}: H5 has {len(gene_names)} features but scanpy read {adata_full.n_vars}"
        )
    adata_full.var["tenx_gene_name"] = gene_names
    adata_full.var["tenx_feature_type"] = feature_types
    mask = adata_full.var["tenx_feature_type"].astype(str).eq("Gene Expression").to_numpy()
    if not mask.any():
        raise ValueError(f"{h5_path} contains no Gene Expression features.")
    adata_gex = adata_full[:, mask].copy()
    symbols = adata_gex.var["tenx_gene_name"].astype(str).str.strip().str.upper()
    if symbols.eq("").any():
        raise ValueError(f"{h5_path} contains empty gene symbols.")
    adata_gex.var_names = pd.Index(symbols, name="gene_symbol")
    adata_gex.obs_names = pd.Index(
        [clean_text(value).upper() for value in adata_gex.obs_names],
        dtype="object",
    )
    return adata_full, adata_gex


def collapse_duplicate_gene_symbols(adata_input):
    ad, _h5py, _uc, _sc, sp = import_ucell_stack()
    symbols = pd.Index(adata_input.var_names.astype(str), dtype="object")
    if not symbols.has_duplicates:
        return adata_input
    codes, unique_symbols = pd.factorize(symbols, sort=True)
    column_map = sp.csr_matrix(
        (
            np.ones(len(codes), dtype=np.float32),
            (np.arange(len(codes), dtype=np.int64), codes.astype(np.int64)),
        ),
        shape=(len(codes), len(unique_symbols)),
    )
    matrix = adata_input.X
    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(matrix)
    collapsed = matrix.tocsr() @ column_map
    return ad.AnnData(
        X=collapsed.tocsr(),
        obs=adata_input.obs.copy(),
        var=pd.DataFrame(index=pd.Index(unique_symbols.astype(str), name="gene_symbol")),
    )


def compute_ucell_scores_compat(adata_input, signatures: Dict[str, List[str]], args) -> None:
    _ad, _h5py, uc, _sc, _sp = import_ucell_stack()
    function = uc.compute_ucell_scores
    parameters = inspect.signature(function).parameters
    kwargs = {"signatures": signatures}
    optional = {
        "layer": "counts",
        "max_rank": args.ucell_max_rank,
        "chunk_size": args.ucell_chunk_size,
        "n_jobs": args.ucell_n_jobs,
        "missing_genes": "skip",
    }
    for key, value in optional.items():
        if key in parameters:
            kwargs[key] = value
    function(adata_input, **kwargs)


def find_ucell_column(columns: Iterable[object], pathway_id: str) -> str:
    names = list(map(str, columns))
    lookup = {name.lower(): name for name in names}
    for candidate in (f"{pathway_id}_UCell", pathway_id):
        if candidate in names:
            return candidate
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise KeyError(f"Cannot find pyUCell output column for {pathway_id}; columns={names[-30:]}")


def compute_ucell_table(
    matched_instances: pd.DataFrame,
    h5_files: Mapping[str, Path],
    signatures: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _ad, _h5py, _uc, _sc, sp = import_ucell_stack()
    unique_cells = (
        matched_instances[["donor", "barcode", "Tcellstate"]]
        .drop_duplicates()
        .sort_values(["donor", "barcode"])
        .reset_index(drop=True)
    )
    state_conflicts = unique_cells.groupby(["donor", "barcode"])["Tcellstate"].nunique()
    if state_conflicts.gt(1).any():
        raise ValueError("One donor/barcode has multiple four-state annotations.")
    unique_cells = unique_cells.drop_duplicates(["donor", "barcode"]).reset_index(drop=True)

    score_blocks = []
    barcode_qc_rows = []
    overlap_rows = []
    h5_unmatched_rows = []

    for donor in VALID_DONORS:
        donor_cells = unique_cells.loc[unique_cells["donor"].eq(donor)].copy()
        if donor_cells.empty:
            continue
        h5_path = Path(h5_files[donor])
        if not h5_path.exists():
            raise FileNotFoundError(h5_path)
        adata_full, adata_gex = read_10x_h5_gene_expression(h5_path)
        requested = donor_cells["barcode"].drop_duplicates().tolist()
        mapping, methods, unmatched = map_requested_barcodes_to_available(
            requested,
            adata_gex.obs_names.astype(str).tolist(),
        )
        barcode_qc_rows.append(
            {
                "donor": donor,
                "n_h5_cells": int(adata_gex.n_obs),
                "n_requested_barcodes": int(len(requested)),
                "n_exact_matches": int(sum(value == "exact" for value in methods.values())),
                "n_core_unique_matches": int(sum(value == "core_unique" for value in methods.values())),
                "n_unmatched_barcodes": int(len(unmatched)),
            }
        )
        h5_unmatched_rows.extend({"donor": donor, "barcode": barcode} for barcode in unmatched)
        if unmatched:
            raise RuntimeError(
                f"{donor}: {len(unmatched)} matched-state barcodes are absent or ambiguous in H5; "
                f"examples={unmatched[:20]}"
            )

        selected_requested = [barcode for barcode in requested if barcode in mapping]
        selected_h5 = [mapping[barcode] for barcode in selected_requested]
        adata_sub = adata_gex[selected_h5, :].copy()
        adata_sub.obs["instance_barcode"] = selected_requested
        adata_sub.obs["donor"] = donor
        adata_sub = collapse_duplicate_gene_symbols(adata_sub)
        if not sp.issparse(adata_sub.X):
            adata_sub.X = sp.csr_matrix(np.asarray(adata_sub.X))
        else:
            adata_sub.X = adata_sub.X.tocsr()
        adata_sub.layers["counts"] = adata_sub.X.copy()
        observed_genes = set(adata_sub.var_names.astype(str))
        donor_signatures: Dict[str, List[str]] = {}
        for pathway_id, genes in signatures.items():
            present = sorted(observed_genes.intersection(genes))
            overlap_rows.append(
                {
                    "donor": donor,
                    "pathway_id": pathway_id,
                    "n_kegg_genes": int(len(genes)),
                    "n_present_genes": int(len(present)),
                    "present_fraction": len(present) / len(genes) if genes else np.nan,
                }
            )
            if len(present) < args.min_pathway_genes:
                raise RuntimeError(
                    f"{donor}/{pathway_id}: only {len(present)} signature genes are present; "
                    f"minimum={args.min_pathway_genes}"
                )
            donor_signatures[pathway_id] = list(genes)

        compute_ucell_scores_compat(adata_sub, donor_signatures, args)
        column_map = {
            pathway_id: find_ucell_column(adata_sub.obs.columns, pathway_id)
            for pathway_id in donor_signatures
        }
        wide = adata_sub.obs[["instance_barcode"] + list(column_map.values())].copy().rename(
            columns={column: pathway_id for pathway_id, column in column_map.items()}
        )
        long = (
            wide.melt(
                id_vars=["instance_barcode"],
                value_vars=sorted(donor_signatures),
                var_name="pathway_id",
                value_name="_ucell_score",
            )
            .rename(columns={"instance_barcode": "barcode"})
        )
        long.insert(0, "donor", donor)
        long["_ucell_score"] = pd.to_numeric(long["_ucell_score"], errors="raise")
        if long.duplicated(["donor", "barcode", "pathway_id"]).any():
            raise RuntimeError(f"{donor}: duplicated cell/pathway UCell scores.")
        score_blocks.append(long)
        del adata_full, adata_gex, adata_sub

    if not score_blocks:
        raise RuntimeError("No UCell scores were produced.")
    scores = pd.concat(score_blocks, ignore_index=True).merge(
        unique_cells,
        on=["donor", "barcode"],
        how="left",
        validate="many_to_one",
    )
    expected = len(unique_cells) * len(PATHWAY_CATALOG)
    if len(scores) != expected:
        raise RuntimeError(f"UCell score rows={len(scores)}, expected={expected}")
    return (
        scores,
        pd.DataFrame(barcode_qc_rows),
        pd.DataFrame(overlap_rows),
        pd.DataFrame(h5_unmatched_rows, columns=["donor", "barcode"]),
    )


# ---------------------------------------------------------------------------
# Quartile context nodes and edges
# ---------------------------------------------------------------------------

def assign_extreme_levels(
    scores: pd.DataFrame,
    low_quantile: float,
    high_quantile: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0 < low_quantile < high_quantile < 1):
        raise ValueError("Quantiles must satisfy 0 < low < high < 1.")
    blocks = []
    threshold_rows = []
    for (donor, pathway_id), block in scores.groupby(
        ["donor", "pathway_id"], sort=True, dropna=False
    ):
        block = block.sort_values(["_ucell_score", "barcode"], kind="mergesort").copy()
        q_low = float(block["_ucell_score"].quantile(low_quantile))
        q_high = float(block["_ucell_score"].quantile(high_quantile))
        block["pathway_level"] = "middle"
        fallback = False
        if q_low < q_high:
            block.loc[block["_ucell_score"].le(q_low), "pathway_level"] = "low"
            block.loc[block["_ucell_score"].ge(q_high), "pathway_level"] = "high"
        else:
            fallback = True
            n_rows = len(block)
            if n_rows < 4:
                raise RuntimeError(
                    f"{donor}/{pathway_id} has only {n_rows} cells; quartiles cannot be defined."
                )
            ranks = block["_ucell_score"].rank(method="first", ascending=True)
            low_cut = max(int(np.floor(n_rows * low_quantile)), 1)
            high_start = min(max(int(np.floor(n_rows * high_quantile)) + 1, low_cut + 1), n_rows)
            block.loc[ranks.le(low_cut), "pathway_level"] = "low"
            block.loc[ranks.ge(high_start), "pathway_level"] = "high"
        threshold_rows.append(
            {
                "donor": donor,
                "pathway_id": pathway_id,
                "n_cells": int(len(block)),
                "q25": q_low,
                "q75": q_high,
                "fallback_rank_used": fallback,
                "n_low": int(block["pathway_level"].eq("low").sum()),
                "n_middle": int(block["pathway_level"].eq("middle").sum()),
                "n_high": int(block["pathway_level"].eq("high").sum()),
            }
        )
        blocks.append(block)
    result = pd.concat(blocks, ignore_index=True).sort_values(
        ["donor", "barcode", "pathway_id"]
    ).reset_index(drop=True)
    if result.duplicated(["donor", "barcode", "pathway_id"]).any():
        raise RuntimeError("A cell/pathway received multiple quartile assignments.")
    return result, pd.DataFrame(threshold_rows)


def build_context_nodes() -> pd.DataFrame:
    rows = [
        {
            "node_id": make_context_node_id(state, pathway_id, level),
            "node_type": "tcellstate_context",
            "Tcellstate": state,
            "pathway_id": pathway_id,
            "pathway_name": pathway_name,
            "pathway_level": level,
        }
        for state in FOUR_STATES
        for pathway_id, pathway_name in PATHWAY_CATALOG
        for level in ("low", "high")
    ]
    nodes = pd.DataFrame(rows)
    expected = len(FOUR_STATES) * len(PATHWAY_CATALOG) * 2
    if len(nodes) != expected or nodes["node_id"].duplicated().any():
        raise RuntimeError("Context-node Cartesian product validation failed.")
    return nodes


def build_context_edges(
    matched_instances: pd.DataFrame,
    level_table: pd.DataFrame,
    context_nodes: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    extremes = level_table.loc[
        level_table["pathway_level"].isin(["low", "high"]),
        ["donor", "barcode", "Tcellstate", "pathway_id", "pathway_level"],
    ].copy()
    node_lookup = context_nodes[
        ["node_id", "Tcellstate", "pathway_id", "pathway_level"]
    ].rename(columns={"node_id": "dst"})
    extremes = extremes.merge(
        node_lookup,
        on=["Tcellstate", "pathway_id", "pathway_level"],
        how="left",
        validate="many_to_one",
    )
    if extremes["dst"].isna().any():
        raise RuntimeError("Some high/low combinations did not map to context nodes.")
    instance_table = matched_instances[
        ["node_id", "donor", "barcode", "Tcellstate"]
    ].rename(columns={"node_id": "src"})
    expanded = instance_table.merge(
        extremes,
        on=["donor", "barcode", "Tcellstate"],
        how="inner",
        validate="many_to_many",
    )
    if expanded.duplicated(["src", "pathway_id"]).any():
        raise RuntimeError("One TCR instance has multiple levels for the same pathway.")
    audit = expanded[
        ["src", "dst", "donor", "barcode", "Tcellstate", "pathway_id", "pathway_level"]
    ].sort_values(["donor", "barcode", "src", "pathway_id"]).reset_index(drop=True)
    edges = audit[["src", "dst"]].copy()
    edges["edge_type"] = "tcr_instance_to_tcellstate_context"
    edges = edges[["src", "dst", "edge_type"]]
    if edges.duplicated().any():
        raise RuntimeError("Duplicated TCR-instance/context edges detected.")
    return edges, audit



def validate_graph_tables(tables: Mapping[str, pd.DataFrame]) -> Dict[str, object]:
    node_table_names = (
        "tcr_sequence_nodes",
        "tcr_instance_nodes",
        "peptide_nodes",
        "vdj_gene_nodes",
        "tcellstate_context_nodes",
    )
    edge_table_names = (
        "tcr_sequence_to_instance_edges",
        "tcr_sequence_to_vdj_gene_edges",
        "tcr_instance_to_tcellstate_context_edges",
        "train_edges",
        "val_edges",
        "test_edges",
    )

    node_blocks = []
    node_counts: Dict[str, int] = {}
    for name in node_table_names:
        frame = tables[name]
        require_columns(frame, ("node_id",), name)
        if frame["node_id"].astype(str).str.strip().eq("").any():
            raise RuntimeError(f"{name} contains empty node_id values.")
        if frame["node_id"].duplicated().any():
            raise RuntimeError(f"{name} contains duplicated node_id values.")
        block = frame[["node_id"]].copy()
        block["source_table"] = name
        node_blocks.append(block)
        node_counts[name] = int(len(frame))

    all_nodes = pd.concat(node_blocks, ignore_index=True)
    duplicated_global = all_nodes.loc[all_nodes["node_id"].duplicated(keep=False)]
    if not duplicated_global.empty:
        raise RuntimeError(
            "Node IDs collide across node types; examples="
            + duplicated_global.head(20).to_json(orient="records")
        )
    node_ids = set(all_nodes["node_id"].astype(str))

    edge_counts: Dict[str, int] = {}
    for name in edge_table_names:
        frame = tables[name]
        require_columns(frame, ("src", "dst", "edge_type"), name)
        if frame.duplicated().any():
            raise RuntimeError(f"{name} contains duplicated rows.")
        missing_src = sorted(set(frame["src"].astype(str)) - node_ids)
        missing_dst = sorted(set(frame["dst"].astype(str)) - node_ids)
        if missing_src or missing_dst:
            raise RuntimeError(
                f"{name} contains missing endpoints: "
                f"missing_src={missing_src[:10]}, missing_dst={missing_dst[:10]}"
            )
        edge_counts[name] = int(len(frame))

    instances = tables["tcr_instance_nodes"]
    instance_metadata = instances[
        ["node_id", "donor", "barcode", "cdr3a", "cdr3b", "tcr_sequence_id"]
    ].rename(columns={"node_id": "src"})
    sequence_sets: Dict[str, set] = {}
    pair_sets: Dict[str, set] = {}
    for split in ("train", "val", "test"):
        edges = tables[f"{split}_edges"]
        merged = edges.merge(
            instance_metadata,
            on="src",
            how="left",
            validate="many_to_one",
            suffixes=("_edge", "_node"),
        )
        for column in ("donor", "barcode", "cdr3a", "cdr3b", "tcr_sequence_id"):
            left = merged[f"{column}_edge"].astype(str)
            right = merged[f"{column}_node"].astype(str)
            if not left.eq(right).all():
                raise RuntimeError(f"{split}_edges metadata disagrees with TCR instance nodes: {column}")
        sequence_sets[split] = set(edges["tcr_sequence_id"].astype(str))
        pair_sets[split] = set(zip(edges["src"].astype(str), edges["dst"].astype(str)))

    overlap_summary: Dict[str, int] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        sequence_overlap = len(sequence_sets[left] & sequence_sets[right])
        pair_overlap = len(pair_sets[left] & pair_sets[right])
        overlap_summary[f"{left}_{right}_tcr_sequence_overlap"] = sequence_overlap
        overlap_summary[f"{left}_{right}_pair_overlap"] = pair_overlap
        if sequence_overlap or pair_overlap:
            raise RuntimeError(
                f"Split leakage detected between {left} and {right}: "
                f"sequence_overlap={sequence_overlap}, pair_overlap={pair_overlap}"
            )

    context_nodes = tables["tcellstate_context_nodes"]
    expected_context = len(FOUR_STATES) * len(PATHWAY_CATALOG) * 2
    if len(context_nodes) != expected_context:
        raise RuntimeError(
            f"Context-node count={len(context_nodes)}, expected={expected_context}"
        )
    forbidden_score_columns = {
        "score",
        "weight",
        "ucell_score",
        "_ucell_score",
        "pathway_score",
    }
    for name in node_table_names + edge_table_names:
        bad = forbidden_score_columns.intersection(tables[name].columns)
        if bad:
            raise RuntimeError(f"{name} unexpectedly exports continuous score columns: {sorted(bad)}")

    return {
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "split_overlap": overlap_summary,
        "all_edge_endpoints_resolved": True,
        "global_node_ids_unique": True,
        "continuous_ucell_score_in_graph": False,
    }


def write_graph_outputs(
    outdir: Path,
    tables: Mapping[str, pd.DataFrame],
    annotations: pd.DataFrame,
    match_audit: pd.DataFrame,
    high_low_audit: pd.DataFrame,
    unmatched_instances: pd.DataFrame,
    vdj_match_audit: pd.DataFrame,
    unmatched_vdj: pd.DataFrame,
    binding: pd.DataFrame,
    threshold_table: pd.DataFrame,
    write_combined: bool,
) -> None:
    filename_map = {
        "tcr_sequence_nodes": "tcr_sequence_nodes.csv",
        "tcr_instance_nodes": "tcr_instance_nodes.csv",
        "peptide_nodes": "peptide_nodes.csv",
        "vdj_gene_nodes": "vdj_gene_nodes.csv",
        "tcellstate_context_nodes": "tcellstate_context_nodes.csv",
        "tcr_sequence_to_instance_edges": "tcr_sequence_to_instance_edges.csv",
        "tcr_sequence_to_vdj_gene_edges": "tcr_sequence_to_vdj_gene_edges.csv",
        "tcr_instance_to_tcellstate_context_edges": "tcr_instance_to_tcellstate_context_edges.csv",
        "train_edges": "train_edges.csv",
        "val_edges": "val_edges.csv",
        "test_edges": "test_edges.csv",
    }
    for table_name, filename in filename_map.items():
        tables[table_name].to_csv(outdir / filename, index=False)

    annotations.to_csv(outdir / "selected_cell_4state_annotations.csv", index=False)
    match_audit.to_csv(outdir / "tcr_instance_4state_ucell_match_audit.csv", index=False)
    high_low_audit.to_csv(
        outdir / "tcr_instance_to_tcellstate_context_high_low_audit.csv",
        index=False,
    )
    unmatched_instances.to_csv(outdir / "unmatched_tcr_instances.csv", index=False)
    vdj_match_audit.to_csv(outdir / "vdj_contig_match_audit.csv", index=False)
    unmatched_vdj.to_csv(outdir / "unmatched_vdj_tcr_instances.csv", index=False)
    binding[
        ["donor", "barcode", "cdr3a", "cdr3b", "peptide", "label", "split"]
    ].to_csv(outdir / "binding_split_audit.csv", index=False)
    threshold_table.to_csv(outdir / "ucell_threshold_audit.csv", index=False)

    if write_combined:
        pd.concat(
            [
                tables["tcr_sequence_nodes"],
                tables["tcr_instance_nodes"],
                tables["peptide_nodes"],
                tables["vdj_gene_nodes"],
                tables["tcellstate_context_nodes"],
            ],
            ignore_index=True,
            sort=False,
        ).to_csv(outdir / "all_nodes.csv", index=False)
        pd.concat(
            [
                tables["tcr_sequence_to_instance_edges"],
                tables["tcr_sequence_to_vdj_gene_edges"],
                tables["tcr_instance_to_tcellstate_context_edges"],
            ],
            ignore_index=True,
            sort=False,
        ).drop_duplicates().to_csv(outdir / "all_context_edges.csv", index=False)


def package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def build_summary(
    args: argparse.Namespace,
    outdir: Path,
    binding_summary: Dict[str, object],
    annotation_summary: Dict[str, object],
    validation: Dict[str, object],
    annotations: pd.DataFrame,
    match_audit: pd.DataFrame,
    matched_instances: pd.DataFrame,
    vdj_match_audit: pd.DataFrame,
    unmatched_vdj: pd.DataFrame,
    scores: pd.DataFrame,
    threshold_table: pd.DataFrame,
    high_low_audit: pd.DataFrame,
    barcode_qc: pd.DataFrame,
    gene_overlap_qc: pd.DataFrame,
    kegg_payload: Dict[str, object],
    kegg_cache_path: Path,
) -> Dict[str, object]:
    unmatched_state = int(match_audit["match_status"].ne("matched").sum())
    context_level_counts = (
        high_low_audit["pathway_level"].value_counts().sort_index().astype(int).to_dict()
    )
    return {
        "status": "success",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_directory": str(outdir),
        "input_and_split": binding_summary,
        "graph_design": {
            "inference_only": bool(getattr(args, "inference_only", False)),
            "tcr_instance_identity": "donor + full_barcode + cdr3a + cdr3b",
            "tcr_sequence_identity": "cdr3a + cdr3b",
            "full_barcode_match_first": True,
            "core_fallback_only_when_unique_within_donor": True,
            "primary_states": list(FOUR_STATES),
            "pathways": [pathway_id for pathway_id, _name in PATHWAY_CATALOG],
            "context_identity": "Tcellstate + pathway_id + pathway_level",
            "low_quantile": float(args.low_quantile),
            "high_quantile": float(args.high_quantile),
            "middle_cells_have_context_edge": False,
            "continuous_ucell_score_written_to_graph": False,
            "vdj_enabled": not args.skip_vdj,
        },
        "annotation": annotation_summary,
        "counts": {
            **validation["node_counts"],
            **validation["edge_counts"],
            "n_annotation_rows": int(len(annotations)),
            "n_matched_state_instances": int(len(matched_instances)),
            "n_unmatched_state_instances": unmatched_state,
            "n_unmatched_vdj_instances": int(len(unmatched_vdj)),
            "n_vdj_audit_rows": int(len(vdj_match_audit)),
            "n_unique_scored_cells": int(
                scores[["donor", "barcode"]].drop_duplicates().shape[0]
            ),
            "n_internal_cell_pathway_scores": int(len(scores)),
            "n_threshold_groups": int(len(threshold_table)),
            "n_threshold_rank_fallback_groups": int(
                threshold_table["fallback_rank_used"].astype(bool).sum()
            ),
            "n_output_context_edges": int(len(high_low_audit)),
        },
        "context_edge_counts_by_level": context_level_counts,
        "context_edge_counts_by_state": (
            high_low_audit["Tcellstate"].value_counts().sort_index().astype(int).to_dict()
        ),
        "barcode_matching_by_donor": barcode_qc.to_dict("records"),
        "kegg_gene_overlap_by_donor": gene_overlap_qc.to_dict("records"),
        "kegg_signature_source": kegg_payload.get("source", "unknown"),
        "kegg_release_info": kegg_payload.get("release_info"),
        "kegg_signature_cache": str(kegg_cache_path),
        "validation": validation,
        "versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scanpy": package_version("scanpy"),
            "anndata": package_version("anndata"),
            "harmonypy": package_version("harmonypy"),
            "pyucell": package_version("pyucell"),
        },
    }


def main() -> None:
    args = build_parser().parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)
    outdir = ensure_output_directory(args.outdir, args.overwrite)

    h5_overrides = parse_donor_path_items(args.h5, "--h5")
    vdj_overrides = parse_donor_path_items(args.vdj_contig, "--vdj-contig")
    h5_files = discover_h5_files(args.h5_dir.expanduser().resolve(), h5_overrides)
    vdj_files = discover_vdj_files(args.vdj_dir.expanduser().resolve(), vdj_overrides)

    print("[1/8] Loading and validating TCR-peptide binding data")
    binding, binding_summary = load_binding_inputs(args)
    required_donors = sorted(set(binding["donor"]))
    for donor in required_donors:
        if not h5_files[donor].exists():
            raise FileNotFoundError(f"Missing expression H5 for {donor}: {h5_files[donor]}")

    print("[2/8] Building TCR instance, TCR sequence, peptide nodes and binding edges")
    base = build_base_graph(binding)

    print("[3/8] Building VDJ gene nodes and edges")
    vdj_nodes, vdj_edges, vdj_match_audit, unmatched_vdj = build_vdj_graph(
        base["tcr_instance_nodes"],
        vdj_files,
        args.skip_vdj,
    )

    print("[4/8] Obtaining four-state cell annotations")
    if args.state_annotations is None:
        annotations, annotation_summary = compute_four_state_annotations(
            binding,
            h5_files,
            args,
        )
    else:
        annotations, annotation_summary = load_existing_state_annotations(
            args.state_annotations
        )
    match_audit, matched_instances = match_instances_to_states(
        base["tcr_instance_nodes"],
        annotations,
    )
    unmatched_state = match_audit.loc[match_audit["match_status"].ne("matched")].copy()
    unmatched_fraction = len(unmatched_state) / len(match_audit) if len(match_audit) else 0.0
    if unmatched_fraction > args.max_unmatched_state_fraction:
        raise RuntimeError(
            f"Unmatched four-state instance fraction={unmatched_fraction:.4%}, "
            f"limit={args.max_unmatched_state_fraction:.4%}"
        )

    print("[5/8] Loading or fetching the 11 KEGG pathway signatures")
    signatures, kegg_payload, kegg_cache_path = load_or_fetch_kegg_signatures(args, outdir)

    print("[6/8] Computing UCell scores from raw Gene Expression counts")
    scores, barcode_qc, gene_overlap_qc, h5_unmatched = compute_ucell_table(
        matched_instances,
        h5_files,
        signatures,
        args,
    )
    level_table, threshold_table = assign_extreme_levels(
        scores,
        args.low_quantile,
        args.high_quantile,
    )
    context_nodes = build_context_nodes()
    context_edges, high_low_audit = build_context_edges(
        matched_instances,
        level_table,
        context_nodes,
    )

    h5_unmatched_for_output = h5_unmatched.copy()
    if h5_unmatched_for_output.empty:
        h5_unmatched_for_output = pd.DataFrame(columns=["donor", "barcode"])
    h5_unmatched_for_output["node_id"] = ""
    h5_unmatched_for_output["match_status"] = "unmatched_h5"
    h5_unmatched_for_output["match_method"] = ""
    h5_unmatched_for_output["Tcellstate"] = ""
    h5_unmatched_for_output["matched_barcode_keys"] = ""
    h5_unmatched_for_output["matched_annotation_rows"] = ""
    h5_unmatched_for_output["unmatched_stage"] = "10x_h5"
    unmatched_state_output = unmatched_state.assign(unmatched_stage="four_state_annotation")
    unmatched_instances = pd.concat(
        [
            unmatched_state_output[
                [
                    "node_id",
                    "donor",
                    "barcode",
                    "match_status",
                    "match_method",
                    "Tcellstate",
                    "matched_barcode_keys",
                    "matched_annotation_rows",
                    "unmatched_stage",
                ]
            ],
            h5_unmatched_for_output[
                [
                    "node_id",
                    "donor",
                    "barcode",
                    "match_status",
                    "match_method",
                    "Tcellstate",
                    "matched_barcode_keys",
                    "matched_annotation_rows",
                    "unmatched_stage",
                ]
            ],
        ],
        ignore_index=True,
    )

    tables = {
        **base,
        "vdj_gene_nodes": vdj_nodes,
        "tcr_sequence_to_vdj_gene_edges": vdj_edges,
        "tcellstate_context_nodes": context_nodes,
        "tcr_instance_to_tcellstate_context_edges": context_edges,
    }

    print("[7/8] Validating global node IDs, endpoints and split leakage")
    validation = validate_graph_tables(tables)

    print("[8/8] Writing graph bundle and audit files")
    write_graph_outputs(
        outdir,
        tables,
        annotations,
        match_audit,
        high_low_audit,
        unmatched_instances,
        vdj_match_audit,
        unmatched_vdj,
        binding,
        threshold_table,
        args.write_combined,
    )
    summary = build_summary(
        args,
        outdir,
        binding_summary,
        annotation_summary,
        validation,
        annotations,
        match_audit,
        matched_instances,
        vdj_match_audit,
        unmatched_vdj,
        scores,
        threshold_table,
        high_low_audit,
        barcode_qc,
        gene_overlap_qc,
        kegg_payload,
        kegg_cache_path,
    )
    for summary_name in (
        "build_complete_graph_summary.json",
        "build_4state_ucell_high_low_summary.json",
    ):
        with open(outdir / summary_name, "w", encoding="utf-8") as handle:
            json.dump(json_safe(summary), handle, ensure_ascii=False, indent=2)

    print("Complete graph built successfully.")
    if args.inference_only:
        print("Inference-only graph: labels are absent; all rows are in test_edges.csv.")
    print("Output directory:", outdir)
    print("Node counts:", validation["node_counts"])
    print("Edge counts:", validation["edge_counts"])
    print("Unmatched state instances:", int(len(unmatched_state)))
    print("Unmatched VDJ instances:", int(len(unmatched_vdj)))


if __name__ == "__main__":
    main()
