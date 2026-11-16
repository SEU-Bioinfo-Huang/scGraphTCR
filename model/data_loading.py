"""Input tables, tokenization, encoder checkpoint specifications, and shared utilities."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from sklearn.model_selection import GroupKFold
    try:
        from sklearn.model_selection import StratifiedGroupKFold
    except Exception:
        StratifiedGroupKFold = None
    SKLEARN_SPLIT_AVAILABLE = True
except Exception:
    GroupKFold = None
    StratifiedGroupKFold = None
    SKLEARN_SPLIT_AVAILABLE = False

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
PAD_ID = 0
MASK_ID = 1
CLS_ID = 2
UNK_ID = 3
SELF_AA_TO_ID = {aa: i + 4 for i, aa in enumerate(AA_ALPHABET)}
SELF_VOCAB_SIZE = 24
DEFAULT_FIVE_FOLD_DATA_DIR = os.path.join(
    "results",
    "5_fold",
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def amp_autocast(args, device: torch.device):
    if not (args.amp and device.type == "cuda"):
        return nullcontext()
    try:
        return torch.amp.autocast("cuda")
    except Exception:
        return torch.cuda.amp.autocast()


def make_grad_scaler(args, device: torch.device):
    enabled = bool(args.amp and device.type == "cuda")
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def safe_torch_load(path: str, map_location):
    """Load tensor checkpoints across PyTorch versions.

    The first call uses weights_only=True. A second call is allowed for local
    checkpoints that contain argparse.Namespace or another harmless metadata
    object not accepted by the restricted unpickler.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except Exception as first_error:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
        except Exception:
            raise first_error


def clean_seq(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip().upper()
    if s in {"", "NAN", "NONE", "NULL", "NA"}:
        return ""
    invalid = sorted(set(s) - set(AA_ALPHABET + "X"))
    if invalid:
        raise ValueError(
            "Self Transformer tokenizer accepts the 20 standard amino acids "
            f"and X; sequence={s!r} contains invalid symbols={invalid}"
        )
    return s


def encode_seq(seq: object, max_len: int) -> List[int]:
    """Encode one sequence exactly as in the self-supervised pretraining.

    The first token is [CLS]. Biological residues occupy the remaining
    max_len - 1 positions. [MASK] is used only during MLM pretraining and is
    never inserted during Stage-2 inference/training.
    """
    if max_len < 2:
        raise ValueError(
            "Self Transformer max sequence length must be at least 2 "
            "because position 0 is reserved for [CLS]."
        )
    s = clean_seq(seq)[: max_len - 1]
    ids = [CLS_ID] + [SELF_AA_TO_ID.get(aa, UNK_ID) for aa in s]
    ids += [PAD_ID] * (max_len - len(ids))
    return ids


def require_columns(df: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}; existing columns={list(df.columns)}")


def read_csv_required(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def find_split_file(base_dir: str, split: str) -> str:
    candidates = [
        os.path.join(base_dir, f"{split}.csv"),
        os.path.join(base_dir, f"{split}_edges.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot find {split}.csv or {split}_edges.csv under {base_dir}")


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=float), -50, 50)))


def binary_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    prob = sigmoid_np(logits)
    out: Dict[str, float] = {}

    if SKLEARN_AVAILABLE and len(np.unique(y_true)) > 1:
        try:
            out["auc"] = float(roc_auc_score(y_true, prob))
        except Exception:
            out["auc"] = float("nan")
        try:
            out["aupr"] = float(average_precision_score(y_true, prob))
        except Exception:
            out["aupr"] = float("nan")
    else:
        out["auc"] = float("nan")
        out["aupr"] = float("nan")

    pred = (prob >= 0.5).astype(int)
    if SKLEARN_AVAILABLE:
        out["acc@0.5"] = float(accuracy_score(y_true, pred))
        out["f1@0.5"] = float(f1_score(y_true, pred, zero_division=0))
        out["precision@0.5"] = float(precision_score(y_true, pred, zero_division=0))
        out["recall@0.5"] = float(recall_score(y_true, pred, zero_division=0))
        try:
            out["mcc@0.5"] = float(matthews_corrcoef(y_true, pred))
        except Exception:
            out["mcc@0.5"] = float("nan")

        best_f1 = -1.0
        best_thr = 0.5
        for threshold in np.linspace(0.01, 0.99, 99):
            p = (prob >= threshold).astype(int)
            score = float(f1_score(y_true, p, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_thr = float(threshold)
        p = (prob >= best_thr).astype(int)
        out["best_threshold"] = best_thr
        out["f1@best"] = best_f1
        out["acc@best"] = float(accuracy_score(y_true, p))
        out["precision@best"] = float(precision_score(y_true, p, zero_division=0))
        out["recall@best"] = float(recall_score(y_true, p, zero_division=0))
        try:
            out["mcc@best"] = float(matthews_corrcoef(y_true, p))
        except Exception:
            out["mcc@best"] = float("nan")
    else:
        out.update({
            "acc@0.5": float((pred == y_true).mean()),
            "f1@0.5": float("nan"),
            "precision@0.5": float("nan"),
            "recall@0.5": float("nan"),
            "mcc@0.5": float("nan"),
            "best_threshold": 0.5,
            "f1@best": float("nan"),
            "acc@best": float((pred == y_true).mean()),
            "precision@best": float("nan"),
            "recall@best": float("nan"),
            "mcc@best": float("nan"),
        })
    return out


def format_metrics(metrics: Dict[str, float], keys: Sequence[str] = ("auc", "aupr", "f1@0.5", "mcc@0.5")) -> str:
    parts = []
    for key in keys:
        value = metrics.get(key, float("nan"))
        if isinstance(value, float) and math.isnan(value):
            parts.append(f"{key}=nan")
        else:
            parts.append(f"{key}={float(value):.4f}")
    return " ".join(parts)


def parse_csv_list(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def split_unique_sequences(sequences: Sequence[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    unique = sorted({clean_seq(x) for x in sequences if clean_seq(x)})
    if not unique:
        raise ValueError("No valid sequences were found for unsupervised pretraining")
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    if len(unique) == 1:
        return unique, unique
    n_val = max(1, int(round(len(unique) * val_fraction)))
    n_val = min(n_val, len(unique) - 1)
    return unique[n_val:], unique[:n_val]


def load_unlabeled_sequence_file(path: str) -> List[str]:
    if not path:
        return []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".txt", ".tsv"}:
        if suffix == ".txt":
            with open(path, "r", encoding="utf-8") as handle:
                return [clean_seq(line) for line in handle if clean_seq(line)]
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
    preferred = ["sequence", "seq", "cdr3", "cdr3a", "cdr3b", "peptide"]
    columns = [c for c in preferred if c in df.columns]
    if not columns:
        if df.shape[1] != 1:
            raise ValueError(f"Cannot identify a sequence column in {path}; columns={list(df.columns)}")
        columns = [df.columns[0]]
    sequences: List[str] = []
    for col in columns:
        sequences.extend(clean_seq(x) for x in df[col].values if clean_seq(x))
    return sequences


@dataclass
class SelfTransformerEncoderSpec:
    name: str
    checkpoint_path: str
    state_dict: Dict[str, torch.Tensor]
    checkpoint_metadata: Dict[str, object]
    vocab_size: int
    max_len: int
    embed_dim: int
    hidden_dim: int
    feedforward_dim: int
    num_layers: int
    n_heads: int
    dropout: float
    has_input_layer_norm: bool
    has_final_norm: bool
    has_output_projection: bool
    norm_first: bool


@dataclass
class SelfTransformerEncoderBundle:
    tcr: SelfTransformerEncoderSpec
    peptide: SelfTransformerEncoderSpec


def _tensor_state_dict(candidate: object) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(candidate, nn.Module):
        return dict(candidate.state_dict())
    if isinstance(candidate, dict) and candidate and all(
        torch.is_tensor(value) for value in candidate.values()
    ):
        return dict(candidate)
    return None


def _extract_state_dict(checkpoint: object, path: str) -> Dict[str, torch.Tensor]:
    state = _tensor_state_dict(checkpoint)
    if state is None and isinstance(checkpoint, dict):
        for key in (
            "state_dict",
            "model_state_dict",
            "model",
            "encoder",
            "sequence_encoder",
            "encoder_state_dict",
        ):
            state = _tensor_state_dict(checkpoint.get(key))
            if state is not None:
                break
    if state is None:
        keys = list(checkpoint.keys())[:30] if isinstance(checkpoint, dict) else []
        raise ValueError(
            f"Cannot locate a tensor state_dict in self-pretraining checkpoint {path}; "
            f"object_type={type(checkpoint)} top_level_keys={keys}"
        )
    return {str(key): value.detach().cpu() for key, value in state.items()}


def _select_encoder_prefix(
    state: Dict[str, torch.Tensor],
    name: str,
    path: str,
) -> Tuple[str, str]:
    """Select one encoder and report its token-embedding key convention.

    Supported conventions include both the earlier generated names
    ``token_emb``/``pos_emb``/``encoder`` and the names used by the user's
    actual checkpoints: ``token_embedding``/``position_embedding``/``transformer``.
    """
    token_suffixes = ("token_emb.weight", "token_embedding.weight")
    candidates: List[Tuple[str, str]] = []
    for key in state:
        for suffix in token_suffixes:
            if key.endswith(suffix):
                candidates.append((key, suffix))
                break
    if not candidates:
        raise ValueError(
            f"Checkpoint {path} has neither token_emb.weight nor "
            f"token_embedding.weight; examples={list(state)[:30]}"
        )

    if len(candidates) == 1:
        selected, suffix = candidates[0]
    else:
        lowered = name.lower()
        preferred = [
            item for item in candidates
            if lowered in item[0].lower()
            or (lowered == "peptide" and "pep" in item[0].lower())
        ]
        if len(preferred) == 1:
            selected, suffix = preferred[0]
        else:
            exact = [item for item in candidates if item[0] == item[1]]
            if len(exact) == 1:
                selected, suffix = exact[0]
            else:
                raise ValueError(
                    f"Checkpoint {path} contains multiple sequence encoders "
                    f"{[item[0] for item in candidates]}; cannot uniquely "
                    f"select the {name} encoder."
                )
    return selected[: -len(suffix)], suffix


def _normalize_self_encoder_state(
    raw_state: Dict[str, torch.Tensor],
    name: str,
    path: str,
) -> Dict[str, torch.Tensor]:
    prefix, _ = _select_encoder_prefix(raw_state, name, path)
    normalized: Dict[str, torch.Tensor] = {}

    root_aliases = (
        ("token_embedding.", "token_emb."),
        ("position_embedding.", "pos_emb."),
        ("transformer.", "encoder."),
        ("token_emb.", "token_emb."),
        ("pos_emb.", "pos_emb."),
        ("encoder.", "encoder."),
        ("input_layer_norm.", "input_layer_norm."),
        ("out_proj.", "out_proj."),
    )

    for key, value in raw_state.items():
        if not key.startswith(prefix):
            continue
        local_key = key[len(prefix):]
        for source_root, target_root in root_aliases:
            if local_key.startswith(source_root):
                normalized[target_root + local_key[len(source_root):]] = value
                break

    if not normalized:
        raise ValueError(f"No compatible self Transformer weights found in {path}")
    return normalized


def _checkpoint_metadata(checkpoint: object) -> Dict[str, object]:
    metadata: Dict[str, object] = {}
    if not isinstance(checkpoint, dict):
        return metadata
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            continue
        if isinstance(value, argparse.Namespace):
            metadata[key] = vars(value)
        elif isinstance(value, (dict, list, tuple, str, int, float, bool, type(None))):
            metadata[key] = value
    return metadata


def _find_metadata_value(metadata: Dict[str, object], names: Sequence[str]) -> Optional[object]:
    wanted = {name.lower() for name in names}
    queue: List[object] = [metadata]
    visited = set()
    while queue:
        current = queue.pop(0)
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, argparse.Namespace):
            current = vars(current)
        if isinstance(current, dict):
            for key, value in current.items():
                if str(key).lower() in wanted:
                    return value
                if isinstance(value, (dict, argparse.Namespace)):
                    queue.append(value)
    return None


def _infer_transformer_architecture(
    state: Dict[str, torch.Tensor],
    path: str,
) -> Tuple[int, int, int, int, int, int, bool, bool, bool]:
    required = [
        "token_emb.weight",
        "pos_emb.weight",
        "encoder.layers.0.self_attn.in_proj_weight",
        "encoder.layers.0.linear1.weight",
    ]
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError(
            f"Checkpoint {path} is not compatible with the expected Transformer "
            f"encoder; missing={missing}; examples={list(state)[:30]}"
        )

    vocab_size, embed_dim = map(int, state["token_emb.weight"].shape)
    max_len, pos_dim = map(int, state["pos_emb.weight"].shape)
    if pos_dim != embed_dim:
        raise ValueError(
            f"token embedding and position embedding dimensions disagree in {path}: "
            f"{embed_dim} vs {pos_dim}"
        )
    feedforward_dim = int(state["encoder.layers.0.linear1.weight"].shape[0])

    has_output_projection = "out_proj.1.weight" in state
    hidden_dim = (
        int(state["out_proj.1.weight"].shape[0])
        if has_output_projection
        else embed_dim
    )
    has_input_layer_norm = (
        "input_layer_norm.weight" in state
        and "input_layer_norm.bias" in state
    )
    has_final_norm = (
        "encoder.norm.weight" in state
        and "encoder.norm.bias" in state
    )

    layer_indices: List[int] = []
    for key in state:
        match = re.fullmatch(r"encoder\.layers\.(\d+)\.self_attn\.in_proj_weight", key)
        if match:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        raise ValueError(f"Cannot infer Transformer layer count from {path}")
    num_layers = max(layer_indices) + 1

    expected_layers = set(range(num_layers))
    observed_layers = set(layer_indices)
    if observed_layers != expected_layers:
        raise ValueError(
            f"Non-contiguous Transformer layers in {path}: observed={sorted(observed_layers)}"
        )
    return (
        vocab_size,
        max_len,
        embed_dim,
        hidden_dim,
        feedforward_dim,
        num_layers,
        has_input_layer_norm,
        has_final_norm,
        has_output_projection,
    )


def _load_self_transformer_spec(
    name: str,
    checkpoint_path: str,
    fallback_n_heads: int,
    dropout: float,
) -> SelfTransformerEncoderSpec:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = safe_torch_load(checkpoint_path, map_location="cpu")
    raw_state = _extract_state_dict(checkpoint, checkpoint_path)
    state = _normalize_self_encoder_state(raw_state, name, checkpoint_path)
    metadata = _checkpoint_metadata(checkpoint)
    (
        vocab_size,
        max_len,
        embed_dim,
        hidden_dim,
        feedforward_dim,
        num_layers,
        has_input_layer_norm,
        has_final_norm,
        has_output_projection,
    ) = _infer_transformer_architecture(state, checkpoint_path)

    metadata_heads = _find_metadata_value(
        metadata,
        ("n_heads", "num_heads", "transformer_heads", "attention_heads"),
    )
    n_heads = int(metadata_heads) if metadata_heads is not None else int(fallback_n_heads)

    metadata_dropout = _find_metadata_value(
        metadata,
        ("dropout", "encoder_dropout", "transformer_dropout"),
    )
    resolved_dropout = (
        float(metadata_dropout) if metadata_dropout is not None else float(dropout)
    )
    metadata_norm_first = _find_metadata_value(metadata, ("norm_first",))
    norm_first = (
        bool(metadata_norm_first) if metadata_norm_first is not None else True
    )

    if n_heads <= 0 or embed_dim % n_heads != 0:
        raise ValueError(
            f"Invalid attention-head count for {name}: embed_dim={embed_dim}, "
            f"n_heads={n_heads}. Supply the pretraining value with --self_n_heads."
        )
    if vocab_size != SELF_VOCAB_SIZE:
        raise ValueError(
            f"{name} checkpoint vocabulary size is {vocab_size}, but the "
            f"self-pretraining tokenizer requires {SELF_VOCAB_SIZE} tokens "
            "([PAD], [CLS], [MASK], [UNK] + 20 amino acids)."
        )

    print(
        f"[{now()}] Loaded self-pretrained {name} Transformer: "
        f"checkpoint={os.path.abspath(checkpoint_path)} vocab={vocab_size} "
        f"max_len={max_len} embed={embed_dim} hidden={hidden_dim} "
        f"ff={feedforward_dim} heads={n_heads} layers={num_layers} "
        f"input_norm={has_input_layer_norm} final_norm={has_final_norm} "
        f"output_projection={has_output_projection} norm_first={norm_first} "
        f"dropout={resolved_dropout:g}"
    )
    return SelfTransformerEncoderSpec(
        name=name,
        checkpoint_path=os.path.abspath(checkpoint_path),
        state_dict=state,
        checkpoint_metadata=metadata,
        vocab_size=vocab_size,
        max_len=max_len,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        feedforward_dim=feedforward_dim,
        num_layers=num_layers,
        n_heads=n_heads,
        dropout=resolved_dropout,
        has_input_layer_norm=has_input_layer_norm,
        has_final_norm=has_final_norm,
        has_output_projection=has_output_projection,
        norm_first=norm_first,
    )


def load_self_transformer_bundle(args) -> SelfTransformerEncoderBundle:
    tcr = _load_self_transformer_spec(
        "tcr",
        args.self_tcr_encoder_path,
        args.self_n_heads,
        args.self_encoder_dropout,
    )
    peptide = _load_self_transformer_spec(
        "peptide",
        args.self_peptide_encoder_path,
        args.self_n_heads,
        args.self_encoder_dropout,
    )
    return SelfTransformerEncoderBundle(tcr=tcr, peptide=peptide)


def resolve_sequence_lengths(args, bundle: SelfTransformerEncoderBundle) -> None:
    requested_tcr = int(args.max_cdr3_len)
    requested_peptide = int(args.max_peptide_len)

    if requested_tcr <= 0:
        args.max_cdr3_len = int(bundle.tcr.max_len)
    elif requested_tcr > bundle.tcr.max_len:
        raise ValueError(
            f"--max_cdr3_len={requested_tcr} exceeds TCR checkpoint positional "
            f"length={bundle.tcr.max_len}"
        )

    if requested_peptide <= 0:
        args.max_peptide_len = int(bundle.peptide.max_len)
    elif requested_peptide > bundle.peptide.max_len:
        raise ValueError(
            f"--max_peptide_len={requested_peptide} exceeds peptide checkpoint positional "
            f"length={bundle.peptide.max_len}"
        )

    print(
        f"[{now()}] Sequence tensor lengths: CDR3B={args.max_cdr3_len} "
        f"peptide={args.max_peptide_len}"
    )


def encoder_mode_settings(mode: str) -> Dict[str, object]:
    aliases = {
        "random": "self_random",
        "self_random": "self_random",
        "teinet_random": "self_random",
        "tcr_only": "self_tcr_only",
        "self_tcr_only": "self_tcr_only",
        "teinet_tcr_only": "self_tcr_only",
        "epi_only": "self_peptide_only",
        "peptide_only": "self_peptide_only",
        "self_epi_only": "self_peptide_only",
        "self_peptide_only": "self_peptide_only",
        "teinet_epi_only": "self_peptide_only",
        "both": "self_both",
        "self": "self_both",
        "self_both": "self_both",
        "teinet": "self_both",
        "teinet_both": "self_both",
    }
    if mode not in aliases:
        raise ValueError(f"Unknown encoder mode={mode}; valid={sorted(aliases)}")
    canonical = aliases[mode]
    return {
        "requested_mode": mode,
        "canonical_mode": canonical,
        "pretrained_tcr": canonical in {"self_tcr_only", "self_both"},
        "pretrained_peptide": canonical in {"self_peptide_only", "self_both"},
        "tcr_input": "cdr3b_only",
        "random_control_uses_pretrained_weights": False,
    }


@dataclass
class PairTensors:
    inst: torch.Tensor
    peptide: torch.Tensor
    label: torch.Tensor
    frame: pd.DataFrame


@dataclass
class GraphData:
    node_table: pd.DataFrame
    node_id_to_idx: Dict[str, int]
    node_type_to_id: Dict[str, int]
    tcr_sequence_node_idx: torch.Tensor
    tcr_sequence_a: torch.Tensor
    tcr_sequence_b: torch.Tensor
    tcr_instance_node_idx: torch.Tensor
    peptide_node_idx: torch.Tensor
    peptide_node_seq: torch.Tensor
    seq_to_inst: torch.Tensor
    vdj_to_seq: torch.Tensor
    state_to_inst: torch.Tensor
    context_node_idx: torch.Tensor
    aux_edges: Dict[str, torch.Tensor]
    context_candidates: Dict[str, torch.Tensor]
    train_pairs: PairTensors
    val_pairs: PairTensors
    test_pairs: PairTensors
    metadata: Dict[str, object]


def _joined_row_text(row: pd.Series) -> str:
    values = []
    for value in row.values.tolist():
        if pd.isna(value):
            continue
        values.append(str(value).strip().upper())
    return " ".join(values)


def infer_beta_gene_type(row: pd.Series) -> Optional[str]:
    """Infer whether a VDJ node is TRBV or TRBJ without assuming fixed columns."""
    text = _joined_row_text(row)
    compact = re.sub(r"[^A-Z0-9-]", "", text)
    if "TRBV" in compact:
        return "TRBV"
    if "TRBJ" in compact:
        return "TRBJ"

    # Fallbacks for tables that store chain and segment type in separate columns.
    has_beta = any(token in text for token in ["BETA", "TRB", "CHAIN_B", "CHAINB"])
    if has_beta:
        if re.search(r"(^|[^A-Z])V([^A-Z]|$)", text) or "VARIABLE" in text:
            return "TRBV"
        if re.search(r"(^|[^A-Z])J([^A-Z]|$)", text) or "JOINING" in text:
            return "TRBJ"
    return None


def _stable_beta_node_id(key_text: str) -> str:
    digest = hashlib.sha1(key_text.encode("utf-8")).hexdigest()[:20]
    return f"tcrbseq_{digest}"


def _make_beta_key(
    cdr3b: str,
    trbv_ids: Sequence[str],
    trbj_ids: Sequence[str],
    beta_node_key: str,
) -> str:
    beta = clean_seq(cdr3b)
    if beta_node_key == "cdr3b":
        return beta
    if beta_node_key == "cdr3b_vj":
        return "\t".join([
            beta,
            ",".join(sorted(set(map(str, trbv_ids)))),
            ",".join(sorted(set(map(str, trbj_ids)))),
        ])
    raise ValueError(f"Unknown beta_node_key={beta_node_key}")


def build_beta_only_node_tables(
    nodes_dir: str,
    beta_node_key: str,
) -> Tuple[
    pd.DataFrame,
    Dict[str, pd.DataFrame],
    Dict[str, int],
    Dict[str, int],
    Dict[str, str],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Rebuild the sequence layer using CDR3B and beta-chain V/J only.

    Returns
    -------
    node_table, tables, node_id_to_idx, node_type_to_id,
    original_sequence_to_beta_id, beta_mapping_audit, beta_vdj_edges
    """
    seq_raw = read_csv_required(os.path.join(nodes_dir, "tcr_sequence_nodes.csv")).copy()
    inst_df = read_csv_required(os.path.join(nodes_dir, "tcr_instance_nodes.csv")).copy()
    pep_df = read_csv_required(os.path.join(nodes_dir, "peptide_nodes.csv")).copy()
    vdj_raw = read_csv_required(os.path.join(nodes_dir, "vdj_gene_nodes.csv")).copy()
    state_df = read_csv_required(os.path.join(nodes_dir, "tcellstate_context_nodes.csv")).copy()
    seq_vdj_raw = read_csv_required(
        os.path.join(nodes_dir, "tcr_sequence_to_vdj_gene_edges.csv")
    ).copy()

    require_columns(seq_raw, ["node_id", "cdr3b"], "tcr_sequence_nodes.csv")
    require_columns(inst_df, ["node_id"], "tcr_instance_nodes.csv")
    require_columns(pep_df, ["node_id", "peptide"], "peptide_nodes.csv")
    require_columns(vdj_raw, ["node_id"], "vdj_gene_nodes.csv")
    require_columns(state_df, ["node_id"], "tcellstate_context_nodes.csv")
    require_columns(seq_vdj_raw, ["src", "dst"], "tcr_sequence_to_vdj_gene_edges.csv")

    for frame in [seq_raw, inst_df, pep_df, vdj_raw, state_df]:
        frame["node_id"] = frame["node_id"].astype(str)
    seq_vdj_raw = seq_vdj_raw[["src", "dst"]].astype(str).drop_duplicates()

    empty_beta = [
        node_id for node_id, value in zip(seq_raw["node_id"], seq_raw["cdr3b"])
        if not clean_seq(value)
    ]
    if empty_beta:
        raise ValueError(
            "tcr_sequence_nodes.csv contains empty CDR3B rows; "
            f"examples={empty_beta[:10]}"
        )

    vdj_raw["beta_gene_type"] = vdj_raw.apply(infer_beta_gene_type, axis=1)
    beta_vdj_df = vdj_raw[vdj_raw["beta_gene_type"].isin(["TRBV", "TRBJ"])].copy()
    if beta_vdj_df.empty:
        raise ValueError(
            "No TRBV/TRBJ nodes were detected in vdj_gene_nodes.csv. "
            "Check gene names or add recognizable TRBV/TRBJ text to the table."
        )
    beta_vdj_ids = set(beta_vdj_df["node_id"].astype(str))
    beta_type_by_id = dict(
        zip(beta_vdj_df["node_id"].astype(str), beta_vdj_df["beta_gene_type"].astype(str))
    )

    beta_edges_old = seq_vdj_raw[seq_vdj_raw["dst"].isin(beta_vdj_ids)].copy()
    linked: Dict[str, Dict[str, List[str]]] = {}
    for old_seq_id, gene_id in beta_edges_old[["src", "dst"]].itertuples(index=False):
        gene_type = beta_type_by_id[str(gene_id)]
        record = linked.setdefault(str(old_seq_id), {"TRBV": [], "TRBJ": []})
        record[gene_type].append(str(gene_id))

    original_to_beta: Dict[str, str] = {}
    audit_rows: List[Dict[str, object]] = []
    grouped_members: Dict[str, List[str]] = {}
    grouped_beta: Dict[str, str] = {}
    grouped_trbv: Dict[str, List[str]] = {}
    grouped_trbj: Dict[str, List[str]] = {}

    for row in seq_raw.itertuples(index=False):
        old_id = str(getattr(row, "node_id"))
        cdr3b = clean_seq(getattr(row, "cdr3b"))
        genes = linked.get(old_id, {"TRBV": [], "TRBJ": []})
        trbv_ids = sorted(set(genes["TRBV"]))
        trbj_ids = sorted(set(genes["TRBJ"]))
        key_text = _make_beta_key(cdr3b, trbv_ids, trbj_ids, beta_node_key)
        beta_id = _stable_beta_node_id(key_text)
        original_to_beta[old_id] = beta_id
        grouped_members.setdefault(beta_id, []).append(old_id)
        grouped_beta[beta_id] = cdr3b
        grouped_trbv.setdefault(beta_id, []).extend(trbv_ids)
        grouped_trbj.setdefault(beta_id, []).extend(trbj_ids)
        audit_rows.append({
            "original_tcr_sequence_id": old_id,
            "beta_sequence_id": beta_id,
            "cdr3b": cdr3b,
            "trbv_ids": ",".join(trbv_ids),
            "trbj_ids": ",".join(trbj_ids),
            "beta_node_key": beta_node_key,
        })

    beta_seq_rows: List[Dict[str, object]] = []
    for beta_id in sorted(grouped_members):
        beta_seq_rows.append({
            "node_id": beta_id,
            "cdr3b": grouped_beta[beta_id],
            "trbv_ids": ",".join(sorted(set(grouped_trbv.get(beta_id, [])))),
            "trbj_ids": ",".join(sorted(set(grouped_trbj.get(beta_id, [])))),
            "n_original_alpha_beta_nodes": len(grouped_members[beta_id]),
            "node_type": "TCR_sequence",
        })
    beta_seq_df = pd.DataFrame(beta_seq_rows)

    # Remap old alpha-beta sequence -> beta V/J edges to the new beta nodes.
    beta_vdj_edges = beta_edges_old.copy()
    beta_vdj_edges["src"] = beta_vdj_edges["src"].map(original_to_beta)
    beta_vdj_edges = beta_vdj_edges.dropna(subset=["src"])
    beta_vdj_edges["gene_type"] = beta_vdj_edges["dst"].map(beta_type_by_id)
    beta_vdj_edges = beta_vdj_edges[["src", "dst", "gene_type"]].drop_duplicates()

    tables: Dict[str, pd.DataFrame] = {
        "TCR_sequence": beta_seq_df.copy(),
        "TCR_instance": inst_df.copy(),
        "peptide": pep_df.copy(),
        "vdj_gene": beta_vdj_df.copy(),
        "tcellstate_context": state_df.copy(),
    }
    specs = [
        ("TCR_sequence", "TCR_sequence"),
        ("TCR_instance", "TCR_instance"),
        ("peptide", "peptide"),
        ("vdj_gene", "vdj_gene"),
        ("tcellstate_context", "tcellstate_context"),
    ]
    rows = []
    for node_type, key in specs:
        frame = tables[key].copy()
        frame["node_id"] = frame["node_id"].astype(str)
        frame["node_type"] = node_type
        tables[key] = frame
        rows.append(frame)

    node_table = pd.concat(rows, ignore_index=True, sort=False)
    if node_table["node_id"].duplicated().any():
        duplicated = node_table.loc[
            node_table["node_id"].duplicated(), "node_id"
        ].head(10).tolist()
        raise ValueError(f"Duplicated node_id values found after beta rebuild: {duplicated}")

    node_id_to_idx = {
        node_id: i for i, node_id in enumerate(node_table["node_id"].astype(str).values)
    }
    node_type_to_id = {node_type: i for i, (node_type, _) in enumerate(specs)}
    audit_df = pd.DataFrame(audit_rows)
    return (
        node_table,
        tables,
        node_id_to_idx,
        node_type_to_id,
        original_to_beta,
        audit_df,
        beta_vdj_edges,
    )


def shuffle_edge_targets(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = df.copy()
    rng = np.random.RandomState(seed)
    values = result["dst"].astype(str).values.copy()
    rng.shuffle(values)
    result["dst"] = values
    return result


def shuffle_beta_vdj_targets_preserve_type(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Shuffle V/J targets independently so TRBV edges remain TRBV and TRBJ remain TRBJ."""
    result = df.copy()
    if "gene_type" not in result.columns:
        return shuffle_edge_targets(result, seed)
    rng = np.random.RandomState(seed)
    for gene_type in ["TRBV", "TRBJ"]:
        mask = result["gene_type"].astype(str).eq(gene_type)
        values = result.loc[mask, "dst"].astype(str).values.copy()
        rng.shuffle(values)
        result.loc[mask, "dst"] = values
    return result


def edge_df_to_index(
    df: pd.DataFrame,
    node_id_to_idx: Dict[str, int],
    device: torch.device,
    name: str,
) -> torch.Tensor:
    require_columns(df, ["src", "dst"], name)
    src_values = df["src"].astype(str).values
    dst_values = df["dst"].astype(str).values
    keep = np.asarray([
        src in node_id_to_idx and dst in node_id_to_idx
        for src, dst in zip(src_values, dst_values)
    ], dtype=bool)
    missing = int((~keep).sum())
    if missing:
        print(f"[{now()}] Graph {name}: dropped {missing} edges with unknown node IDs")
    src = [node_id_to_idx[x] for x in src_values[keep]]
    dst = [node_id_to_idx[x] for x in dst_values[keep]]
    if not src:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def reverse_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.clone()
    return torch.stack([edge_index[1], edge_index[0]], dim=0)


def load_graph_split(nodes_dir: str, split: str) -> pd.DataFrame:
    """Load supervised instance-peptide edges without retaining alpha-chain fields."""
    path = os.path.join(nodes_dir, f"{split}_edges.csv")
    df = read_csv_required(path).copy()
    require_columns(df, ["src", "dst", "label"], path)
    df["src"] = df["src"].astype(str)
    df["dst"] = df["dst"].astype(str)
    df["label"] = df["label"].astype(float)

    # Donor/barcode are retained strictly as split/audit metadata. They are never
    # passed into the model. CDR3A/alpha-chain fields remain excluded.
    keep = [
        column
        for column in [
            "src", "dst", "label", "donor", "barcode", "cdr3b", "peptide"
        ]
        if column in df.columns
    ]
    return df[keep].copy()


def make_pair_tensors(
    df: pd.DataFrame,
    node_id_to_idx: Dict[str, int],
    device: torch.device,
    name: str,
) -> PairTensors:
    missing_inst = sorted(set(df["src"]) - set(node_id_to_idx))
    missing_pep = sorted(set(df["dst"]) - set(node_id_to_idx))
    if missing_inst or missing_pep:
        raise ValueError(
            f"{name}: node IDs missing from node tables; "
            f"instance examples={missing_inst[:5]}, peptide examples={missing_pep[:5]}"
        )
    return PairTensors(
        inst=torch.tensor([node_id_to_idx[x] for x in df["src"]], dtype=torch.long, device=device),
        peptide=torch.tensor([node_id_to_idx[x] for x in df["dst"]], dtype=torch.long, device=device),
        label=torch.tensor(df["label"].values.astype(np.float32), dtype=torch.float32, device=device),
        frame=df.reset_index(drop=True).copy(),
    )


def mode_settings(mode: str) -> Dict[str, object]:

    simple_aliases = {
        "only_seq": "only_seq",
        "only_graph": "only_graph",
        "graph_only": "only_graph",
        "full_no_residue": "full_no_residue",
        # Backward-compatible names. They now resolve to the final
        # no-residue model because the residue graph has been removed.
        "full_residue": "full_no_residue",
        "full": "full_no_residue",
    }
    if mode in simple_aliases:
        canonical = simple_aliases[mode]
        use_sequence = canonical != "only_graph"
        use_graph = canonical in {"only_graph", "full_no_residue"}
        return {
            "requested_mode": mode,
            "canonical_mode": canonical,
            "use_sequence": use_sequence,
            "use_graph": use_graph,
            "use_vdj": use_graph,
            "use_state": use_graph,
            "shuffle_vdj": False,
            "shuffle_state": False,
            "context_integration": (
                "learned_graph_message_passing" if use_graph else "none"
            ),
            "learned_graph_message_passing": use_graph,
            "directed_context_only": False,
            "peptide_bypasses_graph": not use_graph,
            "instance_has_direct_sequence_features": False,
        }

    aliases = {
        "no_graph": "no_graph",
        "seqinst": "no_graph",          
        "vdj": "vdj",
        "seqinst_vdj": "vdj",
        "state": "state",
        "seqinst_state": "state",
        "full": "full",
        "shuffle_vdj": "shuffle_vdj",
        "shuffle_state": "shuffle_state",
        "shuffle_all": "shuffle_all",
    }
    if mode not in aliases:
        valid_modes = sorted(set(aliases) | set(simple_aliases))
        raise ValueError(f"Unknown ablation mode={mode}; valid={valid_modes}")
    canonical = aliases[mode]

    graph_modes = {
        "vdj", "state", "full", "shuffle_vdj", "shuffle_state", "shuffle_all"
    }
    use_vdj = canonical in {
        "vdj", "full", "shuffle_vdj", "shuffle_state", "shuffle_all"
    }
    use_state = canonical in {
        "state", "full", "shuffle_vdj", "shuffle_state", "shuffle_all"
    }
    context_integration = (
        "none" if canonical == "no_graph" else "learned_graph_message_passing"
    )

    return {
        "requested_mode": mode,
        "canonical_mode": canonical,
        "use_vdj": use_vdj,
        "use_state": use_state,
        "use_sequence": True,
        "use_graph": canonical in graph_modes,
        "shuffle_vdj": canonical in {"shuffle_vdj", "shuffle_all"},
        "shuffle_state": canonical in {"shuffle_state", "shuffle_all"},
        "context_integration": context_integration,
        "learned_graph_message_passing": canonical in graph_modes,
        "directed_context_only": canonical in graph_modes,
        "peptide_bypasses_graph": True,
        "instance_has_direct_sequence_features": False,
    }


def _coverage_report(
    seq_to_inst_df: pd.DataFrame,
    pair_frames: Sequence[pd.DataFrame],
    instance_ids: Sequence[str],
) -> Dict[str, int]:
    mapped_instances = set(seq_to_inst_df["dst"].astype(str))
    all_instances = set(map(str, instance_ids))
    supervised_instances = set()
    for frame in pair_frames:
        supervised_instances.update(frame["src"].astype(str).tolist())
    missing_all = all_instances - mapped_instances
    missing_supervised = supervised_instances - mapped_instances
    if missing_supervised:
        raise ValueError(
            "TCR_sequence -> TCR_instance mapping is incomplete for supervised pairs. "
            f"Missing examples={sorted(missing_supervised)[:10]}"
        )
    return {
        "all_instance_nodes": len(all_instances),
        "mapped_instance_nodes": len(all_instances & mapped_instances),
        "missing_instance_nodes": len(missing_all),
        "supervised_instance_nodes": len(supervised_instances),
        "missing_supervised_instance_nodes": len(missing_supervised),
    }


def build_graph_data(
    args,
    mode: str,
    device: torch.device,
    split_frames: Optional[Dict[str, pd.DataFrame]] = None,
    split_metadata: Optional[Dict[str, object]] = None,
) -> GraphData:
    settings = mode_settings(mode)
    (
        node_table,
        tables,
        node_id_to_idx,
        node_type_to_id,
        original_to_beta,
        beta_mapping_audit,
        beta_vdj_edges,
    ) = build_beta_only_node_tables(args.nodes_dir, args.beta_node_key)

    seq_df = tables["TCR_sequence"]
    inst_df = tables["TCR_instance"]
    pep_df = tables["peptide"]
    require_columns(seq_df, ["node_id", "cdr3b"], "rebuilt_beta_sequence_nodes")
    require_columns(inst_df, ["node_id"], "tcr_instance_nodes.csv")
    require_columns(pep_df, ["node_id", "peptide"], "peptide_nodes.csv")

    empty_beta = [i for i, value in enumerate(seq_df["cdr3b"].values) if not clean_seq(value)]
    empty_peptide = [i for i, value in enumerate(pep_df["peptide"].values) if not clean_seq(value)]
    if empty_beta:
        raise ValueError(f"Rebuilt beta sequence nodes contain empty CDR3B rows; examples={empty_beta[:10]}")
    if empty_peptide:
        raise ValueError(f"peptide_nodes.csv contains empty peptide rows; examples={empty_peptide[:10]}")

    beta_truncated = int(sum(len(clean_seq(x)) > args.max_cdr3_len - 1 for x in seq_df["cdr3b"].values))
    peptide_truncated = int(sum(len(clean_seq(x)) > args.max_peptide_len - 1 for x in pep_df["peptide"].values))
    if beta_truncated or peptide_truncated:
        print(
            f"[{now()}] Self Transformer tokenizer truncation: CDR3B={beta_truncated}, "
            f"peptide={peptide_truncated}"
        )

    tcr_sequence_node_idx = torch.tensor(
        [node_id_to_idx[x] for x in seq_df["node_id"].astype(str).values],
        dtype=torch.long,
        device=device,
    )
    # Maintained only for interface compatibility. It is always zero and never
    # passed to a sequence encoder with semantic content.
    tcr_sequence_a = torch.zeros(
        (len(seq_df), args.max_cdr3_len),
        dtype=torch.long,
        device=device,
    )
    tcr_sequence_b = torch.tensor(
        [encode_seq(x, args.max_cdr3_len) for x in seq_df["cdr3b"].values],
        dtype=torch.long,
        device=device,
    )
    tcr_instance_node_idx = torch.tensor(
        [node_id_to_idx[x] for x in inst_df["node_id"].astype(str).values],
        dtype=torch.long,
        device=device,
    )
    peptide_node_idx = torch.tensor(
        [node_id_to_idx[x] for x in pep_df["node_id"].astype(str).values],
        dtype=torch.long,
        device=device,
    )
    peptide_node_seq = torch.tensor(
        [encode_seq(x, args.max_peptide_len) for x in pep_df["peptide"].values],
        dtype=torch.long,
        device=device,
    )

    # Remap original alpha-beta sequence nodes to rebuilt beta-only sequence nodes.
    seqinst_path = os.path.join(args.nodes_dir, "tcr_sequence_to_instance_edges.csv")
    seqinst_raw = read_csv_required(seqinst_path).copy()
    require_columns(seqinst_raw, ["src", "dst"], seqinst_path)
    seqinst_raw = seqinst_raw[["src", "dst"]].astype(str).drop_duplicates()
    unknown_old_seq = sorted(set(seqinst_raw["src"]) - set(original_to_beta))
    if unknown_old_seq:
        raise ValueError(
            "Sequence-to-instance edges reference sequence IDs absent from "
            f"tcr_sequence_nodes.csv; examples={unknown_old_seq[:10]}"
        )
    seqinst_df = seqinst_raw.copy()
    seqinst_df["src"] = seqinst_df["src"].map(original_to_beta)
    seqinst_df = seqinst_df.dropna(subset=["src"])[["src", "dst"]].drop_duplicates()

    beta_sources_per_instance = seqinst_df.groupby("dst")["src"].nunique()
    multi_beta_instances = int((beta_sources_per_instance > 1).sum())
    if multi_beta_instances:
        print(
            f"[{now()}] Beta-only graph: {multi_beta_instances} TCR instances map to "
            "more than one beta sequence node; their sequence messages are mean-aggregated."
        )
    seq_to_inst = edge_df_to_index(seqinst_df, node_id_to_idx, device, "beta_seq_to_inst_mapping")

    if split_frames is None:
        train_df = load_graph_split(args.nodes_dir, "train")
        val_df = load_graph_split(args.nodes_dir, "val")
        test_df = load_graph_split(args.nodes_dir, "test")
    else:
        missing_split_names = [
            name for name in ("train", "val", "test") if name not in split_frames
        ]
        if missing_split_names:
            raise ValueError(
                f"split_frames missing required splits={missing_split_names}"
            )
        train_df = split_frames["train"].copy().reset_index(drop=True)
        val_df = split_frames["val"].copy().reset_index(drop=True)
        test_df = split_frames["test"].copy().reset_index(drop=True)
        for split_name, split_df in [
            ("train", train_df), ("val", val_df), ("test", test_df)
        ]:
            require_columns(split_df, ["src", "dst", "label"], f"{split_name} split")
            split_df["src"] = split_df["src"].astype(str)
            split_df["dst"] = split_df["dst"].astype(str)
            split_df["label"] = split_df["label"].astype(float)
    mapping_coverage = _coverage_report(
        seqinst_df,
        [train_df, val_df, test_df],
        inst_df["node_id"].astype(str).tolist(),
    )

    empty = torch.empty((2, 0), dtype=torch.long, device=device)
    vdj_to_seq = empty.clone()
    state_to_inst = empty.clone()
    aux_edges: Dict[str, torch.Tensor] = {}
    context_candidates: Dict[str, torch.Tensor] = {}
    edge_counts: Dict[str, int] = {"beta_seq_to_inst_mapping": int(seq_to_inst.size(1))}

    if bool(settings["use_vdj"]):
        vdj_df = beta_vdj_edges.copy()
        if bool(settings["shuffle_vdj"]):
            vdj_df = shuffle_beta_vdj_targets_preserve_type(vdj_df, args.seed + 501)
        seq_to_vdj = edge_df_to_index(vdj_df[["src", "dst"]], node_id_to_idx, device, "beta_seq_to_vdj")
        vdj_to_seq = reverse_edge_index(seq_to_vdj)
        aux_edges["vdj"] = seq_to_vdj
        context_candidates["vdj"] = torch.tensor(
            [node_id_to_idx[x] for x in tables["vdj_gene"]["node_id"].astype(str).values],
            dtype=torch.long,
            device=device,
        )
        edge_counts["trbv_trbj_to_beta_seq_directed"] = int(vdj_to_seq.size(1))
        edge_counts["trbv_edges"] = int((vdj_df["gene_type"] == "TRBV").sum())
        edge_counts["trbj_edges"] = int((vdj_df["gene_type"] == "TRBJ").sum())

    if bool(settings["use_state"]):
        path = os.path.join(args.nodes_dir, "tcr_instance_to_tcellstate_context_edges.csv")
        state_df = read_csv_required(path).copy()
        require_columns(state_df, ["src", "dst"], path)
        state_df = state_df[["src", "dst"]].astype(str).drop_duplicates()
        if bool(settings["shuffle_state"]):
            state_df = shuffle_edge_targets(state_df, args.seed + 502)
        inst_to_state = edge_df_to_index(state_df, node_id_to_idx, device, "inst_to_state")
        state_to_inst = reverse_edge_index(inst_to_state)
        aux_edges["state"] = inst_to_state
        context_candidates["state"] = torch.tensor(
            [node_id_to_idx[x] for x in tables["tcellstate_context"]["node_id"].astype(str).values],
            dtype=torch.long,
            device=device,
        )
        edge_counts["state_to_inst_directed"] = int(state_to_inst.size(1))

    context_ids = list(tables["vdj_gene"]["node_id"].astype(str).values) + list(
        tables["tcellstate_context"]["node_id"].astype(str).values
    )
    context_node_idx = torch.tensor(
        [node_id_to_idx[x] for x in context_ids],
        dtype=torch.long,
        device=device,
    )

    train_pairs = make_pair_tensors(train_df, node_id_to_idx, device, "train_edges.csv")
    val_pairs = make_pair_tensors(val_df, node_id_to_idx, device, "val_edges.csv")
    test_pairs = make_pair_tensors(test_df, node_id_to_idx, device, "test_edges.csv")

    missing_trbv_nodes = int(seq_df["trbv_ids"].fillna("").astype(str).eq("").sum())
    missing_trbj_nodes = int(seq_df["trbj_ids"].fillna("").astype(str).eq("").sum())
    original_seq_count = int(beta_mapping_audit["original_tcr_sequence_id"].nunique())
    beta_seq_count = int(seq_df["node_id"].nunique())

    metadata = {
        "mode": mode,
        "settings": settings,
        "n_nodes": len(node_table),
        "node_type_counts": node_table["node_type"].value_counts().to_dict(),
        "directed_edge_counts": edge_counts,
        "mapping_coverage": mapping_coverage,
        "train_pairs": len(train_df),
        "val_pairs": len(val_df),
        "test_pairs": len(test_df),
        "peptide_updated_by_context_graph": False,
        "reverse_hub_aggregation_enabled": False,
        "sequence_backend": "self_pretrained_transformer",
        "tcr_sequence_input": "cdr3b_only",
        "explicit_cdr3a_used": False,
        "alpha_vj_nodes_used": False,
        "retained_vj_node_types": ["TRBV", "TRBJ"],
        "beta_node_key": args.beta_node_key,
        "original_alpha_beta_sequence_nodes": original_seq_count,
        "rebuilt_beta_sequence_nodes": beta_seq_count,
        "collapsed_sequence_nodes": original_seq_count - beta_seq_count,
        "beta_nodes_without_trbv": missing_trbv_nodes,
        "beta_nodes_without_trbj": missing_trbj_nodes,
        "instances_with_multiple_beta_nodes": multi_beta_instances,
        "max_cdr3_tensor_length": args.max_cdr3_len,
        "max_peptide_tensor_length": args.max_peptide_len,
        "truncated_cdr3b_sequences": beta_truncated,
        "truncated_peptide_sequences": peptide_truncated,
        "directed_hierarchy": "TRBV/TRBJ->CDR3B_sequence->TCR_instance<-Tcellstate",
        "context_integration": settings["context_integration"],
        "learned_graph_message_passing": bool(settings["learned_graph_message_passing"]),
        "use_sequence": bool(settings.get("use_sequence", True)),
        "use_graph": bool(
            settings.get(
                "use_graph",
                settings.get("learned_graph_message_passing", False),
            )
        ),
        "final_model": "full_no_residue",
        "residue_graph_removed": True,
        "graph_attention_mode": str(args.graph_attention_mode),
        "graph_attention_relations": [
            "TCR_sequence->TCR_instance",
            "VDJ->TCR_sequence",
            "Tcellstate->TCR_instance",
        ],
    }
    if split_metadata:
        metadata["split_metadata"] = dict(split_metadata)

    # Persist the rebuilt graph audit at the root output directory. These files
    # contain no CDR3A sequence values.
    audit_dir = os.path.join(args.outdir, "beta_graph_audit")
    ensure_dir(audit_dir)
    beta_mapping_audit.to_csv(
        os.path.join(audit_dir, "original_sequence_to_beta_sequence.csv"), index=False
    )
    seq_df.to_csv(os.path.join(audit_dir, "rebuilt_beta_sequence_nodes.csv"), index=False)
    beta_vdj_edges.to_csv(os.path.join(audit_dir, "rebuilt_beta_vdj_edges.csv"), index=False)
    seqinst_df.to_csv(os.path.join(audit_dir, "rebuilt_beta_sequence_to_instance_edges.csv"), index=False)
    with open(os.path.join(audit_dir, "beta_graph_audit.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(
        f"[{now()}] Beta-only graph mode={mode} canonical={settings['canonical_mode']} "
        f"integration={settings['context_integration']} "
        f"beta_key={args.beta_node_key} original_seq={original_seq_count} "
        f"beta_seq={beta_seq_count} nodes={len(node_table)} "
        f"trbv_trbj_edges={vdj_to_seq.size(1)} state_edges={state_to_inst.size(1)} "
        f"train={len(train_df)} val={len(val_df)} test={len(test_df)}"
    )

    return GraphData(
        node_table=node_table,
        node_id_to_idx=node_id_to_idx,
        node_type_to_id=node_type_to_id,
        tcr_sequence_node_idx=tcr_sequence_node_idx,
        tcr_sequence_a=tcr_sequence_a,
        tcr_sequence_b=tcr_sequence_b,
        tcr_instance_node_idx=tcr_instance_node_idx,
        peptide_node_idx=peptide_node_idx,
        peptide_node_seq=peptide_node_seq,
        seq_to_inst=seq_to_inst,
        vdj_to_seq=vdj_to_seq,
        state_to_inst=state_to_inst,
        context_node_idx=context_node_idx,
        aux_edges=aux_edges,
        context_candidates=context_candidates,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        test_pairs=test_pairs,
        metadata=metadata,
    )
