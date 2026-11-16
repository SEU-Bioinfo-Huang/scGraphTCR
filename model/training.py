"""Neural networks, optimization, checkpoint loading, prediction, and attention exports."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_loading import (
    GraphData,
    PAD_ID,
    PairTensors,
    SelfTransformerEncoderBundle,
    SelfTransformerEncoderSpec,
    amp_autocast,
    binary_metrics,
    build_beta_only_node_tables,
    build_graph_data,
    clean_seq,
    encoder_mode_settings,
    ensure_dir,
    format_metrics,
    make_grad_scaler,
    mode_settings,
    now,
    safe_torch_load,
    set_seed,
    sigmoid_np,
)

class InspectableTransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Checkpoint-compatible Transformer layer with optional attention capture.

    The normal path delegates to PyTorch's original implementation. Attention
    weights are collected only when ``capture_attention`` is enabled, which is
    used by the post-training interpretability export and is therefore not part
    of the ordinary training path.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capture_attention = False
        self.last_attention: Optional[torch.Tensor] = None

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not self.capture_attention:
            return super()._sa_block(
                x,
                attn_mask,
                key_padding_mask,
                is_causal=is_causal,
            )

        try:
            attention_output, attention_weights = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
                is_causal=is_causal,
            )
        except TypeError:
            # Compatibility with older PyTorch versions whose MultiheadAttention
            # does not expose the is_causal keyword.
            attention_output, attention_weights = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                average_attn_weights=False,
            )
        self.last_attention = attention_weights.detach()
        return self.dropout1(attention_output)

    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not self.capture_attention:
            return super().forward(
                src,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                is_causal=is_causal,
            )

        # This is the reference TransformerEncoderLayer computation without
        # PyTorch's fused fast path, because that path does not expose weights.
        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x
                + self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                )
            )
            x = self.norm2(x + self._ff_block(x))
        return x

    def clear_attention(self) -> None:
        self.last_attention = None


class SelfSequenceEncoder(nn.Module):
    """Architecture-compatible implementation of the user's Transformer encoder."""

    def __init__(
        self,
        spec: SelfTransformerEncoderSpec,
        expose_attention: bool = False,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(
            spec.vocab_size,
            spec.embed_dim,
            padding_idx=PAD_ID,
        )
        self.pos_emb = nn.Embedding(spec.max_len, spec.embed_dim)
        self.input_layer_norm = (
            nn.LayerNorm(spec.embed_dim)
            if spec.has_input_layer_norm
            else nn.Identity()
        )
        layer_class = (
            InspectableTransformerEncoderLayer
            if expose_attention
            else nn.TransformerEncoderLayer
        )
        encoder_layer = layer_class(
            d_model=spec.embed_dim,
            nhead=spec.n_heads,
            dim_feedforward=spec.feedforward_dim,
            dropout=spec.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=spec.norm_first,
        )
        final_norm = nn.LayerNorm(spec.embed_dim) if spec.has_final_norm else None
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=spec.num_layers,
            norm=final_norm,
        )
        if spec.has_output_projection:
            self.out_proj = nn.Sequential(
                nn.LayerNorm(spec.embed_dim),
                nn.Linear(spec.embed_dim, spec.hidden_dim),
                nn.GELU(),
                nn.Dropout(spec.dropout),
            )
        else:
            self.out_proj = nn.Identity()
        self.spec = spec
        self.expose_attention = bool(expose_attention)

    def _set_attention_capture(self, enabled: bool) -> None:
        if not self.expose_attention:
            if enabled:
                raise RuntimeError(
                    "This Transformer was not constructed with attention exposure. "
                    "Set --save_transformer_attention when creating the model."
                )
            return
        for layer in self.encoder.layers:
            if not isinstance(layer, InspectableTransformerEncoderLayer):
                raise RuntimeError(
                    "Inspectable Transformer layer construction failed."
                )
            layer.capture_attention = bool(enabled)
            if not enabled:
                layer.clear_attention()

    def _collected_attention(self) -> List[torch.Tensor]:
        if not self.expose_attention:
            raise RuntimeError(
                "Attention exposure is disabled for this Transformer."
            )
        collected: List[torch.Tensor] = []
        for layer in self.encoder.layers:
            if not isinstance(layer, InspectableTransformerEncoderLayer):
                raise RuntimeError(
                    "Inspectable Transformer layer construction failed."
                )
            if layer.last_attention is None:
                raise RuntimeError(
                    "Transformer attention was not captured during the forward pass."
                )
            collected.append(layer.last_attention)
        return collected

    def forward(
        self,
        ids: torch.Tensor,
        return_attention: bool = False,
    ):
        if ids.ndim != 2:
            raise ValueError(f"Transformer input must be [batch,length], got {tuple(ids.shape)}")
        batch_size, seq_len = ids.shape
        if seq_len > self.spec.max_len:
            raise ValueError(
                f"{self.spec.name} input length={seq_len} exceeds checkpoint "
                f"positional length={self.spec.max_len}"
            )
        positions = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(batch_size, seq_len)
        pad_mask = ids.eq(PAD_ID)
        self._set_attention_capture(bool(return_attention))
        try:
            x = self.token_emb(ids) + self.pos_emb(positions)
            x = self.input_layer_norm(x)
            x = self.encoder(x, src_key_padding_mask=pad_mask)
            x = self.out_proj(x)
            if return_attention:
                return x, pad_mask, self._collected_attention()
            return x, pad_mask
        finally:
            # Never retain attention tensors after a diagnostic call.
            self._set_attention_capture(False)


def masked_mean_pool(x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).to(dtype=x.dtype).unsqueeze(-1)
    denominator = valid.sum(dim=1).clamp_min(1.0)
    return (x * valid).sum(dim=1) / denominator


class SelfTransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        spec: SelfTransformerEncoderSpec,
        load_pretrained: bool,
        expose_attention: bool = False,
    ):
        super().__init__()
        self.model = SelfSequenceEncoder(
            spec,
            expose_attention=expose_attention,
        )
        self.spec = spec
        self.output_dim = spec.hidden_dim
        self.load_pretrained = bool(load_pretrained)
        if self.load_pretrained:
            incompatible = self.model.load_state_dict(spec.state_dict, strict=True)
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected self Transformer state mismatch for {spec.name}: {incompatible}"
                )

    def encode_tokens(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return token-level Transformer features and the padding mask.

        This does not alter the pretrained encoder. Position 0 is the [CLS] token;
        biological residues begin at position 1. The pooled sequence representation
        is used by the no-residue context model; [CLS] and [PAD] handling remains
        unchanged.
        """
        if ids.ne(PAD_ID).sum(dim=1).eq(0).any():
            raise ValueError(
                f"{self.spec.name} encoder received an empty sequence. "
                "Check CDR3B/peptide node files."
            )
        return self.model(ids)

    def encode_tokens_with_attention(
        self,
        ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Return token features, padding mask, and per-layer/head attention."""
        if not self.model.expose_attention:
            raise RuntimeError(
                "Attention exposure is disabled for this Transformer."
            )
        if ids.ne(PAD_ID).sum(dim=1).eq(0).any():
            raise ValueError(
                f"{self.spec.name} encoder received an empty sequence. "
                "Check CDR3B/peptide node files."
            )
        return self.model(ids, return_attention=True)

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        token_features, pad_mask = self.encode_tokens(ids)
        return masked_mean_pool(token_features, pad_mask)

    def set_trainability(self, mode: str) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        if mode == "frozen":
            return
        if mode in {"last_gru", "last_transformer"}:
            for parameter in self.model.encoder.layers[-1].parameters():
                parameter.requires_grad = True
            for parameter in self.model.out_proj.parameters():
                parameter.requires_grad = True
            return
        if mode == "all":
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            return
        raise ValueError(f"Unknown self Transformer trainability mode={mode}")


class TCRNodeEncoder(nn.Module):
    """Use the self-pretrained TCR Transformer encoder on CDR3 beta only."""

    def __init__(self, sequence_encoder: SelfTransformerSequenceEncoder):
        super().__init__()
        self.sequence_encoder = sequence_encoder

    def forward(self, cdr3a: torch.Tensor, cdr3b: torch.Tensor) -> torch.Tensor:
        del cdr3a  # Explicitly unused: the current graph intentionally uses CDR3 beta only.
        return self.sequence_encoder.encode(cdr3b)

    def encode_tokens(self, cdr3b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequence_encoder.encode_tokens(cdr3b)


class PeptideNodeEncoder(nn.Module):
    def __init__(self, sequence_encoder: SelfTransformerSequenceEncoder):
        super().__init__()
        self.sequence_encoder = sequence_encoder

    def forward(self, peptide: torch.Tensor) -> torch.Tensor:
        return self.sequence_encoder.encode(peptide)

    def encode_tokens(self, peptide: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequence_encoder.encode_tokens(peptide)


def mean_aggregate_global(
    source_features: torch.Tensor,
    edge_index: torch.Tensor,
    n_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate source features to global destination indices by arithmetic mean.

    The function is AMP-safe: aggregation buffers use source_features.dtype and
    any autocast output is converted before index_add_.
    """
    dim = source_features.size(1)
    device = source_features.device
    output = torch.zeros((n_nodes, dim), dtype=source_features.dtype, device=device)
    degree = torch.zeros((n_nodes, 1), dtype=source_features.dtype, device=device)
    if edge_index.numel() == 0:
        return output, degree
    src = edge_index[0]
    dst = edge_index[1]
    messages = source_features[src]
    if messages.dtype != output.dtype:
        messages = messages.to(dtype=output.dtype)
    output.index_add_(0, dst, messages)
    degree.index_add_(
        0,
        dst,
        torch.ones((dst.numel(), 1), dtype=output.dtype, device=device),
    )
    output = output / degree.clamp_min(1.0)
    return output, degree


def grouped_softmax(
    scores: torch.Tensor,
    group_index: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """Apply a numerically stable softmax independently within each group."""
    if scores.ndim != 1 or group_index.ndim != 1:
        raise ValueError("Grouped softmax expects one-dimensional score and group tensors.")
    if scores.numel() != group_index.numel():
        raise ValueError("Grouped softmax score/group lengths do not match.")
    if scores.numel() == 0:
        return scores.clone()

    max_per_group = torch.full(
        (n_groups,),
        torch.finfo(scores.dtype).min,
        dtype=scores.dtype,
        device=scores.device,
    )
    if hasattr(max_per_group, "scatter_reduce_"):
        max_per_group.scatter_reduce_(
            0,
            group_index,
            scores,
            reduce="amax",
            include_self=True,
        )
    else:
        # Compatibility fallback for older PyTorch releases.
        for group in torch.unique(group_index).tolist():
            mask = group_index.eq(int(group))
            max_per_group[int(group)] = scores[mask].max()

    exponent = (scores - max_per_group[group_index]).exp()
    denominator = torch.zeros(
        (n_groups,),
        dtype=scores.dtype,
        device=scores.device,
    )
    denominator.index_add_(0, group_index, exponent)
    return exponent / denominator[group_index].clamp_min(1e-12)


def uniform_group_weights(
    edge_index: torch.Tensor,
    n_groups: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the exact uniform weights represented by mean aggregation."""
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=dtype, device=edge_index.device)
    destination = edge_index[1]
    degree = torch.zeros((n_groups,), dtype=dtype, device=edge_index.device)
    degree.index_add_(
        0,
        destination,
        torch.ones((destination.numel(),), dtype=dtype, device=edge_index.device),
    )
    return 1.0 / degree[destination].clamp_min(1.0)


class RelationAwareEdgeAttention(nn.Module):
    """Destination-normalized attention for one directed graph relation.

    For an edge source ``i`` -> destination ``j``:

        score(i,j) = v^T tanh(W_source h_i + W_target h_j)
        alpha(i,j) = softmax_i(score(i,j)) over incoming edges of j

    The returned attention is an edge-level diagnostic. It is not a binding
    probability and is only used by the opt-in learned graph-attention path.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.source_projection = nn.Linear(dim, dim, bias=False)
        self.target_projection = nn.Linear(dim, dim, bias=False)
        self.score_projection = nn.Linear(dim, 1, bias=False)

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        edge_index: torch.Tensor,
        n_nodes: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = torch.zeros(
            (n_nodes, source_features.size(1)),
            dtype=source_features.dtype,
            device=source_features.device,
        )
        if edge_index.numel() == 0:
            return (
                output,
                torch.zeros(
                    (n_nodes, 1),
                    dtype=source_features.dtype,
                    device=source_features.device,
                ),
                torch.empty(
                    0,
                    dtype=source_features.dtype,
                    device=source_features.device,
                ),
            )

        source = source_features[edge_index[0]]
        target = target_features[edge_index[1]]
        hidden = torch.tanh(
            self.source_projection(source)
            + self.target_projection(target)
        )
        scores = self.score_projection(hidden).squeeze(-1)
        attention = grouped_softmax(scores, edge_index[1], n_nodes)
        # Keep the original message content unchanged; only the neighbor
        # aggregation weights are learned.
        messages = source * attention.unsqueeze(1)
        output.index_add_(0, edge_index[1], messages)

        degree = torch.zeros(
            (n_nodes, 1),
            dtype=source_features.dtype,
            device=source_features.device,
        )
        degree.index_add_(
            0,
            edge_index[1],
            torch.ones(
                (edge_index[1].numel(), 1),
                dtype=source_features.dtype,
                device=source_features.device,
            ),
        )
        return output, degree, attention


@dataclass(frozen=True)
class GateConfiguration:
    """Resolved graph-gate configuration used by one training run."""

    requested: str
    kind: str
    value: float
    label: str


def parse_gate_configuration(spec: str, default_init: float) -> GateConfiguration:
    """Parse ``learnable``, ``learnable_<init>`` or ``fixed_<value>``.

    Examples
    --------
    learnable
    learnable_0.3
    fixed_0.0
    fixed_0.1
    fixed_0.5
    fixed_1.0
    """
    requested = str(spec).strip().lower()
    if requested == "learnable":
        kind = "learnable"
        value = float(default_init)
    elif requested.startswith("learnable_"):
        kind = "learnable"
        try:
            value = float(requested.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid learnable gate specification: {spec!r}") from exc
    elif requested.startswith("fixed_"):
        kind = "fixed"
        try:
            value = float(requested.split("_", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid fixed gate specification: {spec!r}") from exc
    else:
        raise ValueError(
            f"Unknown gate specification={spec!r}; use learnable, "
            "learnable_<initial_value>, or fixed_<value>."
        )

    if kind == "learnable":
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"Learnable gate initialization must be strictly between 0 and 1; got {value}"
            )
    elif not 0.0 <= value <= 1.0:
        raise ValueError(f"Fixed gate value must be in [0,1]; got {value}")

    label = f"{kind}_{value:g}"
    return GateConfiguration(
        requested=requested,
        kind=kind,
        value=value,
        label=label,
    )


def probability_to_logit(value: float, eps: float = 1e-6) -> float:
    clipped = min(max(float(value), eps), 1.0 - eps)
    return float(math.log(clipped / (1.0 - clipped)))


def optimizer_group_lr(
    optimizer: torch.optim.Optimizer,
    group_name: str,
) -> float:
    for group in optimizer.param_groups:
        if group.get("group_name") == group_name:
            return float(group["lr"])
    return float("nan")


class EdgeDecoder(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([left, right, left * right, torch.abs(left - right)], dim=1)
        return self.network(pair).squeeze(1)


class AuxiliaryEdgeDecoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bilinear(source, target).squeeze(1)


class SelfPretrainedGraphModel(nn.Module):
    """Directed hierarchy with the user's self-pretrained Transformer encoders."""

    def __init__(
        self,
        args,
        graph: GraphData,
        self_bundle: SelfTransformerEncoderBundle,
        encoder_mode: str,
        gate_spec: str,
    ):
        super().__init__()
        encoder_settings = encoder_mode_settings(encoder_mode)
        self.gate_config = parse_gate_configuration(gate_spec, args.gate_init)
        self.graph_attention_mode = str(
            getattr(args, "graph_attention_mode", "mean")
        )
        expose_transformer_attention = bool(
            getattr(args, "save_transformer_attention", False)
        )

        tcr_sequence = SelfTransformerSequenceEncoder(
            self_bundle.tcr,
            load_pretrained=bool(encoder_settings["pretrained_tcr"]),
            expose_attention=expose_transformer_attention,
        )
        peptide_sequence = SelfTransformerSequenceEncoder(
            self_bundle.peptide,
            load_pretrained=bool(encoder_settings["pretrained_peptide"]),
            expose_attention=expose_transformer_attention,
        )
        self.encoder_settings = encoder_settings
        self.tcr_encoder = TCRNodeEncoder(tcr_sequence)
        self.peptide_encoder = PeptideNodeEncoder(peptide_sequence)
        self.tcr_bottleneck = nn.Sequential(
            nn.Linear(self_bundle.tcr.hidden_dim, args.graph_dim),
            nn.LayerNorm(args.graph_dim),
            nn.GELU(),
            nn.Dropout(args.bottleneck_dropout),
        )
        self.peptide_bottleneck = nn.Sequential(
            nn.Linear(self_bundle.peptide.hidden_dim, args.graph_dim),
            nn.LayerNorm(args.graph_dim),
            nn.GELU(),
            nn.Dropout(args.bottleneck_dropout),
        )

        self.n_nodes = len(graph.node_table)
        self.graph_dim = args.graph_dim

        self.context_node_embedding = nn.Embedding(self.n_nodes, args.graph_dim)
        nn.init.normal_(self.context_node_embedding.weight, mean=0.0, std=0.02)

        self.vdj_message = nn.Linear(args.graph_dim, args.graph_dim, bias=False)
        self.state_message = nn.Linear(args.graph_dim, args.graph_dim, bias=False)
        if self.graph_attention_mode == "learned":
            self.tcr_instance_attention = RelationAwareEdgeAttention(args.graph_dim)
            self.vdj_edge_attention = RelationAwareEdgeAttention(args.graph_dim)
            self.state_edge_attention = RelationAwareEdgeAttention(args.graph_dim)
        self.vdj_delta_norm = nn.LayerNorm(args.graph_dim)
        self.state_delta_norm = nn.LayerNorm(args.graph_dim)
        self.context_dropout = nn.Dropout(args.graph_dropout)

        if self.gate_config.kind == "learnable":
            initial_logit = probability_to_logit(self.gate_config.value)
            self.vdj_gate_logit = nn.Parameter(
                torch.full((args.graph_dim,), initial_logit, dtype=torch.float32)
            )
            self.state_gate_logit = nn.Parameter(
                torch.full((args.graph_dim,), initial_logit, dtype=torch.float32)
            )
            self.register_buffer("vdj_fixed_gate", torch.empty(0), persistent=False)
            self.register_buffer("state_fixed_gate", torch.empty(0), persistent=False)
        else:
            self.register_parameter("vdj_gate_logit", None)
            self.register_parameter("state_gate_logit", None)
            self.register_buffer(
                "vdj_fixed_gate",
                torch.full((args.graph_dim,), self.gate_config.value, dtype=torch.float32),
            )
            self.register_buffer(
                "state_fixed_gate",
                torch.full((args.graph_dim,), self.gate_config.value, dtype=torch.float32),
            )

        self.tcr_mask_embedding = nn.Parameter(torch.zeros(args.graph_dim))
        self.peptide_mask_embedding = nn.Parameter(torch.zeros(args.graph_dim))
        self.missing_instance_embedding = nn.Parameter(torch.zeros(args.graph_dim))
        nn.init.normal_(self.tcr_mask_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.peptide_mask_embedding, mean=0.0, std=0.02)
        nn.init.normal_(self.missing_instance_embedding, mean=0.0, std=0.02)

        self.edge_decoder = EdgeDecoder(args.graph_dim, args.decoder_dropout)
        self.aux_decoders = nn.ModuleDict({
            relation: AuxiliaryEdgeDecoder(args.graph_dim)
            for relation in graph.aux_edges.keys()
        })
        self._last_graph_diagnostics: Dict[str, float] = {
            "vdj_coverage": 0.0,
            "state_coverage": 0.0,
            "base_sequence_norm_mean": 0.0,
            "base_instance_norm_mean": 0.0,
            "vdj_message_norm_mean": 0.0,
            "state_message_norm_mean": 0.0,
            "vdj_correction_norm_ratio": 0.0,
            "state_correction_norm_ratio": 0.0,
            "graph_total_correction_norm_ratio": 0.0,
            "context_total_correction_norm_ratio": 0.0,
        }
        self._last_graph_attention: Dict[str, object] = {}

    def sequence_parameters(self) -> List[nn.Parameter]:
        return (
            list(self.tcr_encoder.sequence_encoder.parameters())
            + list(self.peptide_encoder.sequence_encoder.parameters())
        )

    def gate_parameters(self) -> List[nn.Parameter]:
        parameters: List[nn.Parameter] = []
        if self.vdj_gate_logit is not None:
            parameters.append(self.vdj_gate_logit)
        if self.state_gate_logit is not None:
            parameters.append(self.state_gate_logit)
        return parameters

    def set_encoder_trainability(self, mode: str) -> None:
        self.tcr_encoder.sequence_encoder.set_trainability(mode)
        self.peptide_encoder.sequence_encoder.set_trainability(mode)

    def _vdj_gate(self) -> torch.Tensor:
        if self.vdj_gate_logit is not None:
            return torch.sigmoid(self.vdj_gate_logit)
        return self.vdj_fixed_gate

    def _state_gate(self) -> torch.Tensor:
        if self.state_gate_logit is not None:
            return torch.sigmoid(self.state_gate_logit)
        return self.state_fixed_gate

    def _encode_tcr_sequence_nodes(
        self,
        graph: GraphData,
        batch_size: int,
    ) -> torch.Tensor:
        pooled_outputs: List[torch.Tensor] = []
        encoder_trainable = any(p.requires_grad for p in self.tcr_encoder.parameters())
        context = nullcontext() if encoder_trainable else torch.no_grad()
        with context:
            for start in range(0, graph.tcr_sequence_node_idx.numel(), batch_size):
                end = min(start + batch_size, graph.tcr_sequence_node_idx.numel())
                token_features, pad_mask = self.tcr_encoder.encode_tokens(
                    graph.tcr_sequence_b[start:end]
                )
                pooled_outputs.append(masked_mean_pool(token_features, pad_mask))
        return torch.cat(pooled_outputs, dim=0)

    def _encode_peptide_nodes(
        self,
        graph: GraphData,
        batch_size: int,
    ) -> torch.Tensor:
        pooled_outputs: List[torch.Tensor] = []
        encoder_trainable = any(p.requires_grad for p in self.peptide_encoder.parameters())
        context = nullcontext() if encoder_trainable else torch.no_grad()
        with context:
            for start in range(0, graph.peptide_node_idx.numel(), batch_size):
                end = min(start + batch_size, graph.peptide_node_idx.numel())
                token_features, pad_mask = self.peptide_encoder.encode_tokens(
                    graph.peptide_node_seq[start:end]
                )
                pooled_outputs.append(masked_mean_pool(token_features, pad_mask))
        return torch.cat(pooled_outputs, dim=0)

    def _apply_node_mask(
        self,
        features: torch.Tensor,
        rate: float,
        replacement: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        if not (self.training and enabled and rate > 0):
            return features
        mask = torch.rand(features.size(0), device=features.device) < rate
        if not mask.any():
            return features
        return torch.where(mask.unsqueeze(1), replacement.unsqueeze(0), features)

    @staticmethod
    def _norm_mean(
        tensor: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        values = torch.linalg.vector_norm(tensor, dim=1)
        if mask is not None:
            mask = mask.reshape(-1).bool()
            if mask.any():
                values = values[mask]
            else:
                return values.new_tensor(0.0)
        if values.numel() == 0:
            return tensor.new_tensor(0.0)
        return values.mean()

    @classmethod
    def _relative_norm_mean(
        cls,
        delta: torch.Tensor,
        base: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        numerator = torch.linalg.vector_norm(delta, dim=1)
        denominator = torch.linalg.vector_norm(base, dim=1).clamp_min(eps)
        values = numerator / denominator
        if mask is not None:
            mask = mask.reshape(-1).bool()
            if mask.any():
                values = values[mask]
            else:
                return values.new_tensor(0.0)
        if values.numel() == 0:
            return delta.new_tensor(0.0)
        return values.mean()

    def encode_graph(self, graph: GraphData, args, apply_feature_mask: bool) -> torch.Tensor:
        """Encode sequence, context, instance, and peptide nodes.

        Context integration is selected by ``graph.metadata["settings"]``:

        ``none``
            Sequence-derived TCR-instance representation only.

        ``learned_graph_message_passing``
            Preserve the original directed graph path: learned VDJ messages update
            CDR3B-sequence nodes before sequence-to-instance transfer, and learned
            state messages update TCR-instance nodes.
        """
        device = graph.tcr_sequence_node_idx.device
        dtype = torch.float32

        settings = graph.metadata["settings"]
        use_sequence = bool(settings.get("use_sequence", True))
        use_graph = bool(
            settings.get(
                "use_graph",
                settings.get("learned_graph_message_passing", False),
            )
        )

        if use_sequence:
            sequence_raw = self._encode_tcr_sequence_nodes(
                graph, args.node_encode_batch_size
            )
            peptide_raw = self._encode_peptide_nodes(
                graph, args.node_encode_batch_size
            )

            sequence_features = self.tcr_bottleneck(sequence_raw)
            peptide_features = self.peptide_bottleneck(peptide_raw)

            sequence_features = self._apply_node_mask(
                sequence_features,
                args.tcr_feature_mask_rate,
                self.tcr_mask_embedding,
                apply_feature_mask,
            )
            peptide_features = self._apply_node_mask(
                peptide_features,
                args.peptide_feature_mask_rate,
                self.peptide_mask_embedding,
                apply_feature_mask,
            )
        else:
            # only_graph uses learned node-ID embeddings as the initial inputs;
            # no Transformer sequence branch is evaluated in this mode.
            sequence_features = self.context_node_embedding(
                graph.tcr_sequence_node_idx
            )
            peptide_features = self.context_node_embedding(graph.peptide_node_idx)

        if sequence_features.dtype != dtype:
            sequence_features = sequence_features.to(dtype=dtype)
        if peptide_features.dtype != dtype:
            peptide_features = peptide_features.to(dtype=dtype)

        base_sequence_features = sequence_features

        # Context nodes are learned categorical embeddings. In graph modes the
        # embeddings are transformed into directed messages.
        context_features = self.context_node_embedding(graph.context_node_idx)
        if context_features.dtype != dtype:
            context_features = context_features.to(dtype=dtype)
        context_global = torch.zeros(
            (self.n_nodes, self.graph_dim),
            dtype=dtype,
            device=device,
        ).index_copy(0, graph.context_node_idx, context_features)

        integration = str(
            settings.get("context_integration", "learned_graph_message_passing")
        )
        use_graph_messages = (
            use_graph
            and integration == "learned_graph_message_passing"
        )

        vdj_correction = torch.zeros_like(base_sequence_features)
        seq_has_vdj = torch.zeros(
            (base_sequence_features.size(0), 1),
            dtype=base_sequence_features.dtype,
            device=device,
        )

        # Pair-independent base sequence features are used as the destination
        # query for optional graph attention. They contain no binding labels.
        base_sequence_query_global = torch.zeros(
            (self.n_nodes, self.graph_dim),
            dtype=base_sequence_features.dtype,
            device=device,
        ).index_copy(0, graph.tcr_sequence_node_idx, base_sequence_features)
        graph_attention_records: Dict[str, Dict[str, object]] = {}

        if bool(settings["use_vdj"]) and graph.vdj_to_seq.numel() > 0:
            if use_graph_messages:
                # Learned VDJ -> TCR_sequence message. No reverse aggregation into
                # VDJ hubs is performed.
                projected_context = self.vdj_message(context_global)
                if self.graph_attention_mode == "learned":
                    vdj_global, vdj_degree, vdj_attention = self.vdj_edge_attention(
                        projected_context,
                        base_sequence_query_global,
                        graph.vdj_to_seq,
                        self.n_nodes,
                    )
                    vdj_attention_type = "learned_relation_attention"
                else:
                    vdj_global, vdj_degree = mean_aggregate_global(
                        projected_context,
                        graph.vdj_to_seq,
                        self.n_nodes,
                    )
                    vdj_attention = uniform_group_weights(
                        graph.vdj_to_seq,
                        self.n_nodes,
                        projected_context.dtype,
                    )
                    vdj_attention_type = "uniform_mean_baseline"
                graph_attention_records["tcr_vdj"] = {
                    "edge_index": graph.vdj_to_seq.detach(),
                    "weights": vdj_attention.detach(),
                    "attention_type": vdj_attention_type,
                }
                seq_delta = self.vdj_delta_norm(
                    vdj_global[graph.tcr_sequence_node_idx]
                )
                if seq_delta.dtype != base_sequence_features.dtype:
                    seq_delta = seq_delta.to(base_sequence_features.dtype)
                seq_has_vdj = vdj_degree[graph.tcr_sequence_node_idx].gt(0).to(
                    seq_delta.dtype
                )
                gate = self._vdj_gate().to(seq_delta.dtype).unsqueeze(0)
                vdj_correction = (
                    gate
                    * self.context_dropout(seq_delta)
                    * seq_has_vdj
                )

        # In graph modes VDJ messages modify sequence nodes before the fixed
        # sequence-to-instance mapping.
        sequence_features = base_sequence_features + vdj_correction

        # Sequence-derived no-context instance baseline.
        base_seq_global = base_sequence_query_global
        base_mean_inst_global, base_inst_degree = mean_aggregate_global(
            base_seq_global,
            graph.seq_to_inst,
            self.n_nodes,
        )
        if self.graph_attention_mode == "learned":
            base_inst_global, _base_attention_degree, base_tcr_attention = (
                self.tcr_instance_attention(
                    base_seq_global,
                    base_mean_inst_global,
                    graph.seq_to_inst,
                    self.n_nodes,
                )
            )
            tcr_attention_type = "learned_relation_attention"
        else:
            base_inst_global = base_mean_inst_global
            base_tcr_attention = uniform_group_weights(
                graph.seq_to_inst,
                self.n_nodes,
                base_seq_global.dtype,
            )
            tcr_attention_type = "uniform_mean_baseline"
        graph_attention_records["tcr_instance_neighbors"] = {
            "edge_index": graph.seq_to_inst.detach(),
            "weights": base_tcr_attention.detach(),
            "attention_type": tcr_attention_type,
        }
        base_instance_features = base_inst_global[graph.tcr_instance_node_idx]

        # Graph-path instance representation. For no_graph this is identical to
        # the base instance representation because vdj_correction=0.
        seq_global = torch.zeros(
            (self.n_nodes, self.graph_dim),
            dtype=sequence_features.dtype,
            device=device,
        ).index_copy(0, graph.tcr_sequence_node_idx, sequence_features)
        if self.graph_attention_mode == "learned":
            inst_global, inst_degree, tcr_attention = self.tcr_instance_attention(
                seq_global,
                base_mean_inst_global,
                graph.seq_to_inst,
                self.n_nodes,
            )
            graph_attention_records["tcr_instance_neighbors"] = {
                "edge_index": graph.seq_to_inst.detach(),
                "weights": tcr_attention.detach(),
                "attention_type": "learned_relation_attention",
            }
        else:
            inst_global, inst_degree = mean_aggregate_global(
                seq_global,
                graph.seq_to_inst,
                self.n_nodes,
            )
            graph_attention_records["tcr_instance_neighbors"] = {
                "edge_index": graph.seq_to_inst.detach(),
                "weights": uniform_group_weights(
                    graph.seq_to_inst,
                    self.n_nodes,
                    seq_global.dtype,
                ),
                "attention_type": "uniform_mean_baseline",
            }
        instance_features = inst_global[graph.tcr_instance_node_idx]

        missing = inst_degree[graph.tcr_instance_node_idx].eq(0)
        base_missing = base_inst_degree[graph.tcr_instance_node_idx].eq(0)
        if missing.any():
            instance_features = torch.where(
                missing,
                self.missing_instance_embedding.unsqueeze(0),
                instance_features,
            )
        if base_missing.any():
            base_instance_features = torch.where(
                base_missing,
                self.missing_instance_embedding.unsqueeze(0),
                base_instance_features,
            )

        state_correction = torch.zeros_like(instance_features)
        inst_has_state = torch.zeros(
            (instance_features.size(0), 1),
            dtype=instance_features.dtype,
            device=device,
        )

        if bool(settings["use_state"]) and graph.state_to_inst.numel() > 0:
            if use_graph_messages:
                # Learned state -> TCR_instance message. No reverse instance-to-
                # state hub averaging is performed.
                projected_context = self.state_message(context_global)
                if self.graph_attention_mode == "learned":
                    state_global, state_degree, state_attention = (
                        self.state_edge_attention(
                            projected_context,
                            base_inst_global,
                            graph.state_to_inst,
                            self.n_nodes,
                        )
                    )
                    state_attention_type = "learned_relation_attention"
                else:
                    state_global, state_degree = mean_aggregate_global(
                        projected_context,
                        graph.state_to_inst,
                        self.n_nodes,
                    )
                    state_attention = uniform_group_weights(
                        graph.state_to_inst,
                        self.n_nodes,
                        projected_context.dtype,
                    )
                    state_attention_type = "uniform_mean_baseline"
                graph_attention_records["tcr_state"] = {
                    "edge_index": graph.state_to_inst.detach(),
                    "weights": state_attention.detach(),
                    "attention_type": state_attention_type,
                }
                state_delta = self.state_delta_norm(
                    state_global[graph.tcr_instance_node_idx]
                )
                if state_delta.dtype != instance_features.dtype:
                    state_delta = state_delta.to(instance_features.dtype)
                inst_has_state = state_degree[graph.tcr_instance_node_idx].gt(0).to(
                    state_delta.dtype
                )
                gate = self._state_gate().to(state_delta.dtype).unsqueeze(0)
                state_correction = (
                    gate
                    * self.context_dropout(state_delta)
                    * inst_has_state
                )

        self._last_graph_attention = graph_attention_records

        if use_graph_messages:
            instance_features = instance_features + state_correction
        else:
            instance_features = base_instance_features

        # Diagnostics are detached and therefore never alter gradients.
        with torch.no_grad():
            vdj_active = seq_has_vdj.reshape(-1).bool()
            state_active = inst_has_state.reshape(-1).bool()
            total_instance_correction = instance_features - base_instance_features

            graph_total_ratio = (
                self._relative_norm_mean(
                    total_instance_correction,
                    base_instance_features,
                )
                if use_graph_messages
                else total_instance_correction.new_tensor(0.0)
            )
            context_total_ratio = self._relative_norm_mean(
                total_instance_correction,
                base_instance_features,
            )

            self._last_graph_diagnostics = {
                "vdj_coverage": float(vdj_active.float().mean().detach().cpu().item())
                if vdj_active.numel() else 0.0,
                "state_coverage": float(state_active.float().mean().detach().cpu().item())
                if state_active.numel() else 0.0,
                "base_sequence_norm_mean": float(
                    self._norm_mean(base_sequence_features).detach().cpu().item()
                ),
                "base_instance_norm_mean": float(
                    self._norm_mean(base_instance_features).detach().cpu().item()
                ),
                "vdj_message_norm_mean": float(
                    self._norm_mean(vdj_correction, vdj_active).detach().cpu().item()
                ),
                "state_message_norm_mean": float(
                    self._norm_mean(state_correction, state_active).detach().cpu().item()
                ),
                "vdj_correction_norm_ratio": float(
                    self._relative_norm_mean(
                        vdj_correction,
                        base_sequence_features,
                        vdj_active,
                    ).detach().cpu().item()
                ),
                "state_correction_norm_ratio": float(
                    self._relative_norm_mean(
                        state_correction,
                        instance_features - state_correction,
                        state_active,
                    ).detach().cpu().item()
                ) if use_graph_messages else 0.0,
                "graph_total_correction_norm_ratio": float(
                    graph_total_ratio.detach().cpu().item()
                ),
                "context_total_correction_norm_ratio": float(
                    context_total_ratio.detach().cpu().item()
                ),
            }

        # Assemble the final addressable tensor once. Peptide features are copied
        # directly from their sequence bottleneck and never context-transformed.
        node_embeddings = torch.zeros(
            (self.n_nodes, self.graph_dim),
            dtype=dtype,
            device=device,
        )
        node_embeddings = node_embeddings.index_copy(
            0, graph.tcr_sequence_node_idx, sequence_features
        )
        node_embeddings = node_embeddings.index_copy(
            0, graph.tcr_instance_node_idx, instance_features
        )
        node_embeddings = node_embeddings.index_copy(
            0, graph.peptide_node_idx, peptide_features
        )
        node_embeddings = node_embeddings.index_copy(
            0, graph.context_node_idx, context_features
        )
        return node_embeddings

    def binding_logits(
        self,
        node_embeddings: torch.Tensor,
        pairs: PairTensors,
    ) -> torch.Tensor:
        """Decode TCR-instance/peptide pairs without residue corrections."""
        return self.edge_decoder(
            node_embeddings[pairs.inst],
            node_embeddings[pairs.peptide],
        )

    def gate_statistics(self) -> Dict[str, object]:
        vdj = self._vdj_gate().detach()
        state = self._state_gate().detach()
        return {
            "gate_requested": self.gate_config.requested,
            "gate_kind": self.gate_config.kind,
            "gate_label": self.gate_config.label,
            "gate_initial_or_fixed_value": self.gate_config.value,
            "vdj_gate_mean": float(vdj.mean().cpu().item()),
            "vdj_gate_min": float(vdj.min().cpu().item()),
            "vdj_gate_max": float(vdj.max().cpu().item()),
            "state_gate_mean": float(state.mean().cpu().item()),
            "state_gate_min": float(state.min().cpu().item()),
            "state_gate_max": float(state.max().cpu().item()),
        }

    def graph_diagnostics(self) -> Dict[str, float]:
        return dict(self._last_graph_diagnostics)

    def graph_attention_records(self) -> Dict[str, Dict[str, object]]:
        """Return the most recent pair-independent graph edge attention."""
        return dict(self._last_graph_attention)


def sample_auxiliary_loss(
    model: SelfPretrainedGraphModel,
    node_embeddings: torch.Tensor,
    graph: GraphData,
    relation: str,
    max_positive_edges: int,
) -> torch.Tensor:
    if relation not in graph.aux_edges or relation not in model.aux_decoders:
        return node_embeddings.sum() * 0.0
    positive_edges = graph.aux_edges[relation]
    if positive_edges.numel() == 0:
        return node_embeddings.sum() * 0.0

    n_edges = positive_edges.size(1)
    if max_positive_edges > 0 and n_edges > max_positive_edges:
        selection = torch.randperm(n_edges, device=positive_edges.device)[:max_positive_edges]
        positive_edges = positive_edges[:, selection]

    source = positive_edges[0]
    positive_target = positive_edges[1]
    candidates = graph.context_candidates[relation]
    random_index = torch.randint(0, candidates.numel(), (source.numel(),), device=source.device)
    negative_target = candidates[random_index]
    collision = negative_target.eq(positive_target)
    if collision.any() and candidates.numel() > 1:
        replacement_index = torch.randint(
            0,
            candidates.numel(),
            (int(collision.sum()),),
            device=source.device,
        )
        negative_target[collision] = candidates[replacement_index]

    decoder = model.aux_decoders[relation]
    positive_logits = decoder(node_embeddings[source], node_embeddings[positive_target])
    negative_logits = decoder(node_embeddings[source], node_embeddings[negative_target])
    logits = torch.cat([positive_logits, negative_logits], dim=0)
    labels = torch.cat([
        torch.ones_like(positive_logits),
        torch.zeros_like(negative_logits),
    ], dim=0)
    return F.binary_cross_entropy_with_logits(logits, labels)


def make_stage2_optimizer(model: SelfPretrainedGraphModel, args) -> torch.optim.Optimizer:
    sequence_params = model.sequence_parameters()
    gate_params = model.gate_parameters()
    sequence_ids = {id(p) for p in sequence_params}
    gate_ids = {id(p) for p in gate_params}
    graph_params = [
        p for p in model.parameters()
        if id(p) not in sequence_ids and id(p) not in gate_ids
    ]

    parameter_groups: List[Dict[str, object]] = [
        {
            "params": graph_params,
            "lr": args.graph_lr,
            "group_name": "graph",
        },
    ]
    if gate_params:
        parameter_groups.append({
            "params": gate_params,
            "lr": args.graph_lr * args.gate_lr_multiplier,
            "group_name": "gate",
        })
    parameter_groups.append({
        "params": sequence_params,
        "lr": args.encoder_finetune_lr,
        "group_name": "encoder",
    })
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def get_pos_weight(labels: torch.Tensor, requested: float) -> torch.Tensor:
    if requested > 0:
        return torch.tensor(float(requested), device=labels.device)
    positives = float((labels == 1).sum().item())
    negatives = float((labels == 0).sum().item())
    value = negatives / positives if positives > 0 else 1.0
    return torch.tensor(max(value, 1.0), device=labels.device)


@torch.no_grad()
def evaluate_stage2(
    model: SelfPretrainedGraphModel,
    graph: GraphData,
    pairs: PairTensors,
    args,
    criterion: nn.Module,
) -> Tuple[float, Dict[str, float], np.ndarray]:
    model.eval()
    node_embeddings = model.encode_graph(graph, args, apply_feature_mask=False)
    logits = model.binding_logits(node_embeddings, pairs)
    loss = float(criterion(logits, pairs.label).item())
    logits_np = logits.detach().cpu().numpy()
    metrics = binary_metrics(pairs.label.detach().cpu().numpy(), logits_np)
    return loss, metrics, logits_np


def save_predictions(path: str, pairs: PairTensors, logits: np.ndarray) -> None:
    output = pairs.frame.copy()
    output["graph_logit"] = logits
    output["graph_probability"] = sigmoid_np(logits)
    output.to_csv(path, index=False)


def _attention_concentration(
    attention: torch.Tensor,
    query_valid: torch.Tensor,
    key_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return length-normalized concentration and maximum attention.

    Attention rows are softmax-normalized, so their raw row mean is not a useful
    sample-level score. The normalized entropy complement is zero for a nearly
    uniform row and approaches one when the row focuses on a small number of
    valid residues. The second output is the mean row-wise maximum.
    """
    attention = attention.float()
    query_valid = query_valid.bool()
    key_valid = key_valid.bool()
    masked = attention.clamp_min(0.0) * key_valid.unsqueeze(1).to(attention.dtype)
    row_sum = masked.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probabilities = masked / row_sum
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    key_count = key_valid.sum(dim=1)
    log_key_count = torch.log(key_count.clamp_min(2).to(attention.dtype))
    normalized_entropy = entropy / log_key_count.unsqueeze(1)
    concentration = torch.where(
        key_count.gt(1).unsqueeze(1),
        (1.0 - normalized_entropy).clamp(0.0, 1.0),
        torch.zeros_like(normalized_entropy),
    )
    query_weight = query_valid.to(attention.dtype)
    denominator = query_weight.sum(dim=1).clamp_min(1.0)
    concentration_score = (
        concentration * query_weight
    ).sum(dim=1) / denominator
    maximum_score = (
        probabilities.max(dim=-1).values * query_weight
    ).sum(dim=1) / denominator
    return concentration_score, maximum_score


def _empty_interpretability_frame(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


@torch.no_grad()
def export_graph_attention(
    model: SelfPretrainedGraphModel,
    graph: GraphData,
    args,
    mode_outdir: str,
    split_name: str,
) -> Dict[str, object]:
    """Export TCR-neighbor, TCR-VDJ, and TCR-state edge attention."""
    interpretability_dir = os.path.join(
        mode_outdir,
        "interpretability",
        split_name,
    )
    ensure_dir(interpretability_dir)

    # Refresh pair-independent representations and the most recent graph edge
    # weights using the same deterministic path used for evaluation.
    model.eval()
    model.encode_graph(graph, args, apply_feature_mask=False)
    records = model.graph_attention_records()
    node_table = graph.node_table.reset_index(drop=True)

    edge_columns = [
        "attention_relation",
        "attention_type",
        "source_global_idx",
        "source_node_id",
        "source_node_type",
        "destination_global_idx",
        "destination_node_id",
        "destination_node_type",
        "destination_in_degree",
        "attention_weight",
        "attention_rank_within_destination",
    ]
    rows: List[Dict[str, object]] = []

    for relation, record in records.items():
        edge_index = record["edge_index"]
        weights = record["weights"]
        attention_type = str(record["attention_type"])
        if not torch.is_tensor(edge_index) or edge_index.numel() == 0:
            continue
        if not torch.is_tensor(weights) or weights.numel() != edge_index.size(1):
            raise RuntimeError(
                f"Graph attention record {relation} has inconsistent edge weights."
            )

        edge_cpu = edge_index.detach().cpu()
        weight_cpu = weights.detach().float().cpu()
        destination = edge_cpu[1].numpy().astype(int)
        degree = np.bincount(destination, minlength=len(node_table))
        local_rank = (
            pd.Series(weight_cpu.numpy())
            .groupby(pd.Series(destination), sort=False)
            .rank(method="first", ascending=False)
            .astype(int)
            .to_numpy()
        )

        for edge_position, (source_global, destination_global) in enumerate(
            edge_cpu.t().tolist()
        ):
            source_row = node_table.iloc[int(source_global)]
            destination_row = node_table.iloc[int(destination_global)]
            row: Dict[str, object] = {
                "attention_relation": relation,
                "attention_type": attention_type,
                "source_global_idx": int(source_global),
                "source_node_id": str(source_row.get("node_id", source_global)),
                "source_node_type": str(source_row.get("node_type", "")),
                "destination_global_idx": int(destination_global),
                "destination_node_id": str(
                    destination_row.get("node_id", destination_global)
                ),
                "destination_node_type": str(
                    destination_row.get("node_type", "")
                ),
                "destination_in_degree": int(degree[int(destination_global)]),
                "attention_weight": float(weight_cpu[edge_position].item()),
                "attention_rank_within_destination": int(local_rank[edge_position]),
            }
            for column in (
                "cdr3b",
                "peptide",
                "gene_type",
                "pathway_id",
                "level",
                "donor",
                "barcode",
            ):
                if column in source_row.index:
                    row[f"source_{column}"] = source_row[column]
                if column in destination_row.index:
                    row[f"destination_{column}"] = destination_row[column]
            rows.append(row)

    if rows:
        edge_frame = pd.DataFrame(rows)
        extra_columns = [column for column in edge_frame.columns if column not in edge_columns]
        edge_frame = edge_frame[edge_columns + extra_columns]
    else:
        edge_frame = _empty_interpretability_frame(edge_columns)
    edge_path = os.path.join(
        interpretability_dir,
        "graph_attention_edges.csv",
    )
    edge_frame.to_csv(edge_path, index=False)

    summary_rows: List[Dict[str, object]] = []
    if not edge_frame.empty:
        for (relation, attention_type), group in edge_frame.groupby(
            ["attention_relation", "attention_type"], sort=False
        ):
            values = pd.to_numeric(group["attention_weight"], errors="coerce")
            summary_rows.append({
                "attention_relation": relation,
                "attention_type": attention_type,
                "n_edges": int(len(group)),
                "n_destinations": int(group["destination_node_id"].nunique()),
                "mean_attention_weight": float(values.mean()),
                "max_attention_weight": float(values.max()),
                "std_attention_weight": float(values.std(ddof=1))
                if len(group) > 1 else 0.0,
            })
    summary_frame = pd.DataFrame(summary_rows, columns=[
        "attention_relation",
        "attention_type",
        "n_edges",
        "n_destinations",
        "mean_attention_weight",
        "max_attention_weight",
        "std_attention_weight",
    ])
    summary_path = os.path.join(
        interpretability_dir,
        "graph_attention_summary.csv",
    )
    summary_frame.to_csv(summary_path, index=False)

    node_score_specs = {
        "tcr_instance_neighbors": (
            "source_node_id",
            "source_cdr3b",
            "tcr_node_neighbor_attention_scores.csv",
        ),
        "tcr_vdj": (
            "destination_node_id",
            "destination_cdr3b",
            "tcr_vdj_attention_scores.csv",
        ),
        "tcr_state": (
            "destination_node_id",
            None,
            "tcr_state_attention_scores.csv",
        ),
    }
    node_score_files: List[str] = []
    for relation, (node_column, sequence_column, filename) in node_score_specs.items():
        if edge_frame.empty or relation not in set(edge_frame["attention_relation"]):
            score_frame = _empty_interpretability_frame([
                "attention_relation",
                "attention_type",
                "node_id",
                "cdr3b",
                "n_edges",
                "n_other_nodes",
                "mean_attention_weight",
                "max_attention_weight",
                "std_attention_weight",
                "attention_rank",
            ])
        else:
            subset = edge_frame.loc[
                edge_frame["attention_relation"].eq(relation)
            ].copy()
            group_columns = [node_column]
            if sequence_column is not None and sequence_column in subset.columns:
                group_columns.append(sequence_column)
            grouped = subset.groupby(group_columns, dropna=False, sort=False)
            other_column = (
                "destination_node_id"
                if node_column == "source_node_id"
                else "source_node_id"
            )
            score_frame = grouped.agg(
                n_edges=("attention_weight", "count"),
                n_other_nodes=(other_column, "nunique"),
                mean_attention_weight=("attention_weight", "mean"),
                max_attention_weight=("attention_weight", "max"),
                std_attention_weight=("attention_weight", "std"),
            ).reset_index()
            score_frame = score_frame.rename(columns={node_column: "node_id"})
            if sequence_column is not None and sequence_column in score_frame.columns:
                score_frame = score_frame.rename(columns={sequence_column: "cdr3b"})
            elif "cdr3b" not in score_frame.columns:
                score_frame["cdr3b"] = ""
            score_frame["attention_relation"] = relation
            score_frame["attention_type"] = str(
                subset["attention_type"].iloc[0]
            )
            score_frame["std_attention_weight"] = score_frame[
                "std_attention_weight"
            ].fillna(0.0)
            score_frame = score_frame.sort_values(
                "mean_attention_weight",
                ascending=False,
                kind="stable",
            ).reset_index(drop=True)
            score_frame.insert(
                len(score_frame.columns),
                "attention_rank",
                np.arange(1, len(score_frame) + 1, dtype=int),
            )
            score_frame = score_frame[[
                "attention_relation",
                "attention_type",
                "node_id",
                "cdr3b",
                "n_edges",
                "n_other_nodes",
                "mean_attention_weight",
                "max_attention_weight",
                "std_attention_weight",
                "attention_rank",
            ]]
        score_frame.to_csv(os.path.join(interpretability_dir, filename), index=False)
        node_score_files.append(filename)

    return {
        "split": split_name,
        "graph_attention_mode": model.graph_attention_mode,
        "n_edges": int(len(edge_frame)),
        "output_files": [
            "graph_attention_edges.csv",
            "graph_attention_summary.csv",
        ] + node_score_files,
    }


def _transformer_attention_statistics(
    attention: torch.Tensor,
    biological_valid: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize one [batch, heads, length, length] attention tensor."""
    batch_size, n_heads, length, _ = attention.shape
    expanded_valid = biological_valid[:, None, :].expand(
        batch_size,
        n_heads,
        length,
    )
    concentration, maximum = _attention_concentration(
        attention.reshape(batch_size * n_heads, length, length),
        expanded_valid.reshape(batch_size * n_heads, length),
        expanded_valid.reshape(batch_size * n_heads, length),
    )

    attention = attention.float()
    key_mask = biological_valid[:, None, None, :].to(attention.dtype)
    query_mask = biological_valid[:, None, :, None].to(attention.dtype)
    biological_attention = attention * key_mask
    row_sum = biological_attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    biological_attention = biological_attention / row_sum
    query_count = biological_valid.sum(dim=1).clamp_min(1).to(attention.dtype)
    received = (
        biological_attention * query_mask
    ).sum(dim=2) / query_count[:, None, None]
    received = received.mean(dim=1)

    cls_to_residue = torch.zeros(
        (batch_size, length),
        dtype=attention.dtype,
        device=attention.device,
    )
    cls_key_mask = biological_valid[:, None, :].to(attention.dtype)
    cls_rows = attention[:, :, 0, :] * cls_key_mask
    cls_rows = cls_rows / cls_rows.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    cls_to_residue = cls_rows.mean(dim=1)
    return (
        concentration.reshape(batch_size, n_heads).mean(dim=1),
        maximum.reshape(batch_size, n_heads).mean(dim=1),
        received,
        cls_to_residue,
    )


@torch.no_grad()
def export_transformer_attention(
    model: SelfPretrainedGraphModel,
    graph: GraphData,
    args,
    mode_outdir: str,
    split_name: str,
) -> Dict[str, object]:
    """Export sequence-level and residue-level self-Transformer attention."""
    interpretability_dir = os.path.join(
        mode_outdir,
        "interpretability",
        split_name,
    )
    ensure_dir(interpretability_dir)

    sequence_rows: List[Dict[str, object]] = []
    residue_rows: List[Dict[str, object]] = []
    map_candidates: List[Dict[str, object]] = []
    map_limit = max(int(args.interpretability_top_k), 0)

    encoder_inputs = [
        (
            "tcr",
            model.tcr_encoder.sequence_encoder,
            graph.tcr_sequence_b,
            graph.tcr_sequence_node_idx,
            "cdr3b",
        ),
        (
            "peptide",
            model.peptide_encoder.sequence_encoder,
            graph.peptide_node_seq,
            graph.peptide_node_idx,
            "peptide",
        ),
    ]
    node_table = graph.node_table.reset_index(drop=True)
    batch_size = max(int(args.node_encode_batch_size), 1)

    for sequence_type, encoder, sequence_ids, node_indices, sequence_column in encoder_inputs:
        if not encoder.model.expose_attention:
            raise RuntimeError(
                "Transformer attention export was requested, but the encoder "
                "was not constructed with attention exposure."
            )

        sequence_table = node_table.iloc[
            node_indices.detach().cpu().numpy().astype(int)
        ].reset_index(drop=True)
        sequence_strings = [
            clean_seq(value) for value in sequence_table[sequence_column].tolist()
        ]

        for start in range(0, sequence_ids.size(0), batch_size):
            end = min(start + batch_size, sequence_ids.size(0))
            token_features, pad_mask, layer_attention = (
                encoder.encode_tokens_with_attention(sequence_ids[start:end])
            )
            del token_features
            biological_valid = ~pad_mask.bool()
            if biological_valid.size(1) > 0:
                biological_valid = biological_valid.clone()
                biological_valid[:, 0] = False
            if not biological_valid.any(dim=1).all():
                raise RuntimeError(
                    f"{sequence_type} Transformer attention export found an "
                    "empty biological sequence."
                )

            concentration_values: List[torch.Tensor] = []
            maximum_values: List[torch.Tensor] = []
            received_values: List[torch.Tensor] = []
            cls_values: List[torch.Tensor] = []
            for attention in layer_attention:
                concentration, maximum, received, cls_to_residue = (
                    _transformer_attention_statistics(
                        attention,
                        biological_valid,
                    )
                )
                concentration_values.append(concentration)
                maximum_values.append(maximum)
                received_values.append(received)
                cls_values.append(cls_to_residue)

            concentration = torch.stack(concentration_values, dim=0).mean(dim=0)
            maximum = torch.stack(maximum_values, dim=0).mean(dim=0)
            received = torch.stack(received_values, dim=0).mean(dim=0)
            cls_to_residue = torch.stack(cls_values, dim=0).mean(dim=0)
            mean_map = torch.stack(layer_attention, dim=0).mean(dim=0).mean(dim=1)
            mean_map = mean_map.float()
            map_key_mask = biological_valid[:, None, :].to(mean_map.dtype)
            mean_map = mean_map * map_key_mask
            mean_map = mean_map / mean_map.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            for local_index in range(end - start):
                global_local_index = start + local_index
                sequence_text = sequence_strings[global_local_index]
                node_row = sequence_table.iloc[global_local_index]
                node_id = str(node_row.get("node_id", global_local_index))
                sequence_score = float(concentration[local_index].item())
                sequence_rows.append({
                    "sequence_type": sequence_type,
                    "node_id": node_id,
                    "sequence": sequence_text,
                    "sequence_length": len(sequence_text),
                    "transformer_attention_concentration": sequence_score,
                    "transformer_attention_max": float(maximum[local_index].item()),
                    "n_layers": len(layer_attention),
                    "n_heads": int(layer_attention[0].size(1))
                    if layer_attention else 0,
                })

                valid_positions = torch.where(
                    biological_valid[local_index]
                )[0].detach().cpu().tolist()
                for token_position in valid_positions:
                    biological_position = int(token_position)
                    if biological_position <= 0 or biological_position > len(sequence_text):
                        continue
                    residue_rows.append({
                        "sequence_type": sequence_type,
                        "node_id": node_id,
                        "sequence": sequence_text,
                        "position": biological_position,
                        "residue": sequence_text[biological_position - 1],
                        "mean_received_attention": float(
                            received[local_index, biological_position].item()
                        ),
                        "mean_cls_to_residue_attention": float(
                            cls_to_residue[local_index, biological_position].item()
                        ),
                    })

                if map_limit > 0:
                    candidate = {
                        "score": sequence_score,
                        "sequence_type": sequence_type,
                        "node_id": node_id,
                        "sequence": sequence_text,
                        "attention": mean_map[local_index].detach().cpu(),
                        "valid": biological_valid[local_index].detach().cpu(),
                    }
                    if len(map_candidates) < map_limit:
                        map_candidates.append(candidate)
                    else:
                        minimum_index = min(
                            range(len(map_candidates)),
                            key=lambda index: float(map_candidates[index]["score"]),
                        )
                        if sequence_score > float(map_candidates[minimum_index]["score"]):
                            map_candidates[minimum_index] = candidate

    sequence_columns = [
        "sequence_type",
        "node_id",
        "sequence",
        "sequence_length",
        "transformer_attention_concentration",
        "transformer_attention_max",
        "n_layers",
        "n_heads",
    ]
    residue_columns = [
        "sequence_type",
        "node_id",
        "sequence",
        "position",
        "residue",
        "mean_received_attention",
        "mean_cls_to_residue_attention",
    ]
    sequence_frame = pd.DataFrame(sequence_rows, columns=sequence_columns)
    residue_frame = pd.DataFrame(residue_rows, columns=residue_columns)
    sequence_frame = sequence_frame.sort_values(
        "transformer_attention_concentration",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)
    sequence_frame.insert(
        0,
        "attention_rank",
        np.arange(1, len(sequence_frame) + 1, dtype=int),
    )
    sequence_frame.to_csv(
        os.path.join(
            interpretability_dir,
            "transformer_attention_sequence_scores.csv",
        ),
        index=False,
    )
    residue_frame.to_csv(
        os.path.join(
            interpretability_dir,
            "transformer_attention_residue_scores.csv",
        ),
        index=False,
    )

    map_rows: List[Dict[str, object]] = []
    for candidate in sorted(
        map_candidates,
        key=lambda item: float(item["score"]),
        reverse=True,
    ):
        attention = candidate["attention"]
        valid = candidate["valid"]
        valid_positions = torch.where(valid)[0].tolist()
        sequence_text = str(candidate["sequence"])
        for query_position in valid_positions:
            if query_position <= 0 or query_position > len(sequence_text):
                continue
            for key_position in valid_positions:
                if key_position <= 0 or key_position > len(sequence_text):
                    continue
                map_rows.append({
                    "sequence_type": candidate["sequence_type"],
                    "node_id": candidate["node_id"],
                    "sequence": sequence_text,
                    "sequence_attention_concentration": float(candidate["score"]),
                    "query_position": int(query_position),
                    "query_residue": sequence_text[query_position - 1],
                    "key_position": int(key_position),
                    "key_residue": sequence_text[key_position - 1],
                    "mean_attention": float(attention[query_position, key_position].item()),
                })
    map_columns = [
        "sequence_type",
        "node_id",
        "sequence",
        "sequence_attention_concentration",
        "query_position",
        "query_residue",
        "key_position",
        "key_residue",
        "mean_attention",
    ]
    pd.DataFrame(map_rows, columns=map_columns).to_csv(
        os.path.join(
            interpretability_dir,
            "transformer_attention_top_maps.csv",
        ),
        index=False,
    )

    summary = {
        "split": split_name,
        "n_sequences": int(len(sequence_frame)),
        "n_residue_rows": int(len(residue_frame)),
        "top_map_sequences": int(len(map_candidates)),
        "output_files": [
            "transformer_attention_sequence_scores.csv",
            "transformer_attention_residue_scores.csv",
            "transformer_attention_top_maps.csv",
        ],
    }
    with open(
        os.path.join(interpretability_dir, "transformer_attention_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary


def resolve_existing_stage2_checkpoint(
    args,
    mode: str,
    gate_config: GateConfiguration,
    encoder_mode: str,
    split_metadata: Optional[Dict[str, object]],
) -> str:
    """Resolve a checkpoint from the previous fold result directory."""
    checkpoint_root = str(getattr(args, "checkpoint_root", "")).strip()
    if not checkpoint_root:
        raise ValueError(
            "--checkpoint_root is required when --evaluate_only is enabled."
        )
    if not split_metadata or not split_metadata.get("fold"):
        raise ValueError(
            "Evaluation-only mode requires split metadata containing the fold name."
        )

    fold_name = str(split_metadata["fold"])
    encoder_name = str(
        encoder_mode_settings(encoder_mode)["canonical_mode"]
    )
    graph_name = str(mode_settings(mode)["canonical_mode"])
    base_dir = os.path.join(
        os.path.abspath(checkpoint_root),
        fold_name,
        "models",
        encoder_name,
        graph_name,
    )
    candidate_paths = [
        os.path.join(base_dir, gate_config.label, "stage2_best.pt"),
        os.path.join(base_dir, "stage2_best.pt"),
    ]
    requested_mode = str(mode).strip()
    if requested_mode != graph_name:
        requested_dir = os.path.join(
            os.path.abspath(checkpoint_root),
            fold_name,
            "models",
            encoder_name,
            requested_mode,
        )
        candidate_paths.extend([
            os.path.join(requested_dir, gate_config.label, "stage2_best.pt"),
            os.path.join(requested_dir, "stage2_best.pt"),
        ])

    for path in candidate_paths:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Could not find an existing stage2_best.pt. Checked:\n"
        + "\n".join(candidate_paths)
    )


def load_stage2_checkpoint_compat(
    model: SelfPretrainedGraphModel,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, object]:
    """Load old checkpoints while ignoring obsolete residue-module weights."""
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"Invalid Stage-2 checkpoint format: {checkpoint_path}; "
            "expected a dictionary containing the 'model' state_dict."
        )
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Invalid model state_dict in checkpoint: {checkpoint_path}"
        )

    obsolete_keys = [
        key for key in state_dict
        if str(key).startswith("residue_graph.")
    ]
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not str(key).startswith("residue_graph.")
    }
    try:
        model.load_state_dict(filtered_state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"Checkpoint parameters are incompatible with the current model: "
            f"{checkpoint_path}. Only obsolete residue_graph.* keys are ignored."
        ) from exc

    if obsolete_keys:
        print(
            f"[{now()}] Loaded checkpoint with {len(obsolete_keys)} obsolete "
            "residue_graph.* parameters ignored."
        )
    return checkpoint


def _external_checkpoint_base_dir(
    args,
    fold_index: int,
    mode: str,
    encoder_mode: str,
) -> str:
    encoder_name = str(encoder_mode_settings(encoder_mode)["canonical_mode"])
    graph_name = str(mode_settings(mode)["canonical_mode"])
    return os.path.join(
        os.path.abspath(str(args.checkpoint_root)),
        f"{fold_index}fold",
        "models",
        encoder_name,
        graph_name,
    )


def resolve_external_checkpoint(
    args,
    fold_index: int,
    mode: str,
    encoder_mode: str,
    gate_spec: str,
) -> str:
    """Resolve one fold checkpoint for external inference.

    The existing training/evaluation resolver is intentionally left unchanged.
    This resolver additionally supports a gate directory when the external
    graph is evaluated without loading the old split files.
    """
    gate_config = parse_gate_configuration(gate_spec, args.gate_init)
    base_dir = _external_checkpoint_base_dir(
        args, fold_index, mode, encoder_mode
    )
    preferred = [
        os.path.join(base_dir, gate_config.label, "stage2_best.pt"),
        os.path.join(base_dir, "stage2_best.pt"),
    ]
    for path in preferred:
        if os.path.isfile(path):
            return path

    gate_candidates = sorted(
        str(path)
        for path in Path(base_dir).glob("*/stage2_best.pt")
        if path.is_file()
    ) if os.path.isdir(base_dir) else []
    if len(gate_candidates) == 1:
        return gate_candidates[0]
    checked = preferred + gate_candidates
    if not checked:
        checked = [os.path.join(base_dir, "stage2_best.pt")]
    raise FileNotFoundError(
        "Could not resolve an external-inference checkpoint for fold "
        f"{fold_index}. Checked:\n" + "\n".join(checked)
        + "\nPass the same --gate_mode used for the saved checkpoint."
    )


def _checkpoint_saved_args(checkpoint: Dict[str, object]) -> Dict[str, object]:
    saved = checkpoint.get("args")
    if isinstance(saved, dict):
        return saved
    if isinstance(saved, argparse.Namespace):
        return vars(saved)
    return {}


def _external_checkpoint_mode_settings(
    checkpoint: Dict[str, object],
    args,
) -> Tuple[str, str, str]:
    """Read mode, encoder mode and gate specification from one checkpoint."""
    saved_args = _checkpoint_saved_args(checkpoint)
    saved_mode = checkpoint.get("mode") or saved_args.get("single_mode") or args.single_mode
    mode = str(mode_settings(str(saved_mode))["canonical_mode"])

    saved_encoder = checkpoint.get("encoder_mode") or saved_args.get("encoder_mode") or args.encoder_mode
    encoder_mode = str(encoder_mode_settings(str(saved_encoder))["canonical_mode"])

    gate_payload = checkpoint.get("gate_configuration")
    gate_spec = str(args.gate_mode)
    if isinstance(gate_payload, dict):
        gate_spec = str(
            gate_payload.get("label")
            or gate_payload.get("requested")
            or gate_spec
        )
    return mode, encoder_mode, gate_spec


def _copy_saved_architecture_args(
    external_args,
    checkpoint: Dict[str, object],
) -> None:
    """Reuse structural Stage-2 settings stored in the checkpoint when present."""
    saved_args = _checkpoint_saved_args(checkpoint)
    for name in (
        "graph_dim",
        "graph_layers",
        "bottleneck_dropout",
        "graph_dropout",
        "decoder_dropout",
        "self_n_heads",
        "self_encoder_dropout",
        "beta_node_key",
        "gate_lr_multiplier",
        "gate_init",
    ):
        if name in saved_args and saved_args[name] is not None:
            setattr(external_args, name, saved_args[name])
    for name in ("max_cdr3_len", "max_peptide_len"):
        value = saved_args.get(name)
        if value is not None and int(value) > 0:
            setattr(external_args, name, int(value))

    saved_attention_mode = saved_args.get("graph_attention_mode")
    state_dict = checkpoint.get("model")
    if isinstance(state_dict, dict):
        has_learned_relation_attention = any(
            str(key).startswith((
                "tcr_instance_attention.",
                "vdj_edge_attention.",
                "state_edge_attention.",
            ))
            for key in state_dict
        )
        if has_learned_relation_attention:
            saved_attention_mode = "learned"
    if saved_attention_mode in {"mean", "learned"}:
        external_args.graph_attention_mode = str(saved_attention_mode)


def load_external_stage2_checkpoint_compat(
    model: SelfPretrainedGraphModel,
    checkpoint_path: str,
    device: torch.device,
    source_node_id_to_idx: Dict[str, int],
    target_node_id_to_idx: Dict[str, int],
) -> Dict[str, object]:
    """Load a checkpoint while aligning node-indexed embeddings by node_id.

    Sequence and peptide features are generated from their sequences, so only
    the categorical node embedding needs ID-aware alignment when an external
    graph has a different node order or node count. Known node IDs reuse the
    trained row; new IDs start at zero and are reported to the caller.
    """
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"Invalid Stage-2 checkpoint format: {checkpoint_path}; "
            "expected a dictionary containing the 'model' state_dict."
        )
    state_dict = checkpoint["model"]
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Invalid model state_dict in checkpoint: {checkpoint_path}"
        )

    obsolete_keys = {
        key for key in state_dict
        if str(key).startswith("residue_graph.")
    }
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if not str(key).startswith("residue_graph.")
    }
    expected_state = model.state_dict()
    adapted_state: Dict[str, torch.Tensor] = {}
    copied_node_rows = 0
    new_node_ids: List[str] = []

    for key, value in filtered_state_dict.items():
        if key == "context_node_embedding.weight":
            if key not in expected_state or not torch.is_tensor(value):
                raise RuntimeError(
                    f"Checkpoint is missing a compatible {key} tensor: "
                    f"{checkpoint_path}"
                )
            target_weight = torch.zeros_like(expected_state[key])
            source_weight = value.detach().to(
                device=target_weight.device,
                dtype=target_weight.dtype,
            )
            if source_weight.ndim != 2 or target_weight.ndim != 2:
                raise RuntimeError(
                    f"Invalid context embedding dimensions in {checkpoint_path}: "
                    f"source={tuple(source_weight.shape)} target={tuple(target_weight.shape)}"
                )
            if source_weight.size(1) != target_weight.size(1):
                raise RuntimeError(
                    f"Context embedding dimension mismatch in {checkpoint_path}: "
                    f"source={tuple(source_weight.shape)} target={tuple(target_weight.shape)}"
                )
            if source_weight.size(0) != len(source_node_id_to_idx):
                raise RuntimeError(
                    f"Checkpoint/node-template mismatch in {checkpoint_path}: "
                    f"checkpoint_nodes={source_weight.size(0)} "
                    f"template_nodes={len(source_node_id_to_idx)}. "
                    "Use the same training nodes_dir that created the checkpoint."
                )
            for node_id, target_index in target_node_id_to_idx.items():
                source_index = source_node_id_to_idx.get(node_id)
                if source_index is None or source_index >= source_weight.size(0):
                    new_node_ids.append(str(node_id))
                    continue
                if target_index >= target_weight.size(0):
                    raise RuntimeError(
                        f"External node index out of range for {node_id}: "
                        f"{target_index} >= {target_weight.size(0)}"
                    )
                target_weight[target_index] = source_weight[source_index]
                copied_node_rows += 1
            adapted_state[key] = target_weight
            continue

        if key not in expected_state:
            raise RuntimeError(
                f"Checkpoint parameter {key!r} is not present in the external model."
            )
        if tuple(value.shape) != tuple(expected_state[key].shape):
            raise RuntimeError(
                f"Checkpoint parameter shape mismatch for {key!r}: "
                f"checkpoint={tuple(value.shape)} model={tuple(expected_state[key].shape)}"
            )
        adapted_state[key] = value

    missing_keys = sorted(set(expected_state) - set(adapted_state))
    if missing_keys:
        raise RuntimeError(
            f"Checkpoint is missing model parameters for external inference: "
            f"{missing_keys[:20]}"
        )
    model.load_state_dict(adapted_state, strict=True)
    if obsolete_keys:
        print(
            f"[{now()}] External checkpoint ignored {len(obsolete_keys)} obsolete "
            "residue_graph.* parameters."
        )
    if new_node_ids:
        print(
            f"[{now()}] External graph has {len(new_node_ids)} node IDs without "
            "a trained embedding; their categorical rows remain zero."
        )
    checkpoint["external_embedding_alignment"] = {
        "copied_node_rows": int(copied_node_rows),
        "new_node_ids": sorted(set(new_node_ids)),
        "source_node_count": int(len(source_node_id_to_idx)),
        "target_node_count": int(len(target_node_id_to_idx)),
    }
    return checkpoint


def run_external_inference(
    args,
    self_bundle: SelfTransformerEncoderBundle,
    device: torch.device,
) -> pd.DataFrame:
    """Run the saved Stage-2 model on an externally constructed graph."""
    external_nodes_dir = os.path.abspath(str(args.external_graph_dir))
    output_dir = os.path.abspath(str(args.outdir))
    ensure_dir(output_dir)

    # Resolve one checkpoint first so the external graph uses exactly the same
    # structural settings as the saved model whenever checkpoint metadata exists.
    reference_mode = str(mode_settings(args.single_mode)["canonical_mode"])
    reference_encoder = str(
        encoder_mode_settings(args.encoder_mode)["canonical_mode"]
    )
    reference_checkpoint_path = resolve_external_checkpoint(
        args,
        fold_index=1,
        mode=reference_mode,
        encoder_mode=reference_encoder,
        gate_spec=args.gate_mode,
    )
    reference_checkpoint = safe_torch_load(
        reference_checkpoint_path,
        map_location=device,
    )
    if not isinstance(reference_checkpoint, dict):
        raise ValueError(
            f"Invalid reference checkpoint: {reference_checkpoint_path}"
        )
    mode, encoder_mode, gate_spec = _external_checkpoint_mode_settings(
        reference_checkpoint,
        args,
    )
    if mode != "full_no_residue":
        raise ValueError(
            "External inference currently targets the final full_no_residue model; "
            f"checkpoint resolved to mode={mode!r}."
        )

    external_args = copy.copy(args)
    _copy_saved_architecture_args(external_args, reference_checkpoint)
    external_args.nodes_dir = external_nodes_dir
    external_args.outdir = output_dir
    external_args.external_inference = False
    external_args.evaluate_only = False
    external_args.reuse_existing_five_fold_data = False
    external_args.single_mode = mode
    external_args.encoder_mode = encoder_mode
    external_args.gate_mode = gate_spec
    external_args.tcr_feature_mask_rate = 0.0
    external_args.peptide_feature_mask_rate = 0.0

    # The source node table defines the row semantics of the checkpoint's
    # context_node_embedding. External node indices are handled independently.
    (
        _source_node_table,
        _source_tables,
        source_node_id_to_idx,
        _source_type_to_id,
        _source_original_to_beta,
        _source_beta_audit,
        _source_beta_vdj_edges,
    ) = build_beta_only_node_tables(
        os.path.abspath(str(args.nodes_dir)),
        external_args.beta_node_key,
    )

    print(
        f"[{now()}] External graph directory={external_nodes_dir}; "
        f"checkpoint template={os.path.abspath(str(args.nodes_dir))}"
    )
    graph = build_graph_data(
        external_args,
        mode,
        device,
    )
    if graph.test_pairs.frame.empty:
        raise ValueError(
            "External graph has no rows in test_edges.csv. Build it with "
            "--inference-only so all external rows are placed in test_edges.csv."
        )

    output = graph.test_pairs.frame.copy().reset_index(drop=True)
    fold_probabilities: List[np.ndarray] = []
    fold_summaries: List[Dict[str, object]] = []

    for fold_index in range(1, args.group_cv_folds + 1):
        checkpoint_path = resolve_external_checkpoint(
            args,
            fold_index=fold_index,
            mode=mode,
            encoder_mode=encoder_mode,
            gate_spec=gate_spec,
        )
        checkpoint = safe_torch_load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"Invalid checkpoint: {checkpoint_path}")
        fold_mode, fold_encoder, fold_gate = _external_checkpoint_mode_settings(
            checkpoint,
            external_args,
        )
        if (fold_mode, fold_encoder, fold_gate) != (mode, encoder_mode, gate_spec):
            raise ValueError(
                f"Fold {fold_index} checkpoint settings differ from fold 1: "
                f"mode/encoder/gate={(fold_mode, fold_encoder, fold_gate)} vs "
                f"{(mode, encoder_mode, gate_spec)}"
            )

        model = SelfPretrainedGraphModel(
            external_args,
            graph,
            self_bundle,
            encoder_mode,
            gate_spec=gate_spec,
        ).to(device)
        loaded = load_external_stage2_checkpoint_compat(
            model=model,
            checkpoint_path=checkpoint_path,
            device=device,
            source_node_id_to_idx=source_node_id_to_idx,
            target_node_id_to_idx=graph.node_id_to_idx,
        )
        model.eval()
        with torch.no_grad():
            node_embeddings = model.encode_graph(
                graph,
                external_args,
                apply_feature_mask=False,
            )
            logits = model.binding_logits(node_embeddings, graph.test_pairs)
        logits_np = logits.detach().cpu().numpy()
        probability = sigmoid_np(logits_np)
        fold_probabilities.append(probability)
        output[f"fold{fold_index}_graph_logit"] = logits_np
        output[f"fold{fold_index}_graph_probability"] = probability
        fold_summaries.append({
            "fold": f"{fold_index}fold",
            "checkpoint": checkpoint_path,
            "checkpoint_epoch": loaded.get("epoch"),
            "n_rows": int(len(probability)),
            "probability_mean": float(np.mean(probability)),
            "probability_std": float(np.std(probability)),
            "copied_node_rows": loaded.get("external_embedding_alignment", {}).get(
                "copied_node_rows"
            ),
            "new_node_count": len(
                loaded.get("external_embedding_alignment", {}).get("new_node_ids", [])
            ),
        })
        del model, node_embeddings, logits
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probabilities = np.vstack(fold_probabilities)
    output["ensemble_graph_probability_mean"] = probabilities.mean(axis=0)
    output["ensemble_graph_probability_std"] = probabilities.std(axis=0, ddof=0)
    output["ensemble_n_folds"] = int(probabilities.shape[0])
    output_path = os.path.join(output_dir, "external_predictions.csv")
    output.to_csv(output_path, index=False)

    context_mask = graph.node_table["node_type"].astype(str).isin(
        ["vdj_gene", "tcellstate_context"]
    )
    context_ids = graph.node_table.loc[context_mask, "node_id"].astype(str)
    unknown_context_ids = sorted(
        set(context_ids) - set(source_node_id_to_idx)
    )
    summary = {
        "status": "success",
        "mode": mode,
        "encoder_mode": encoder_mode,
        "gate_spec": gate_spec,
        "external_graph_dir": external_nodes_dir,
        "training_node_template_dir": os.path.abspath(str(args.nodes_dir)),
        "n_external_rows": int(len(output)),
        "n_external_instances": int(output["src"].nunique())
        if "src" in output.columns else None,
        "n_external_peptides": int(output["dst"].nunique())
        if "dst" in output.columns else None,
        "n_folds": int(probabilities.shape[0]),
        "labels_used": False,
        "checkpoint_summaries": fold_summaries,
        "unknown_context_node_ids": unknown_context_ids,
        "output_file": output_path,
    }
    with open(
        os.path.join(output_dir, "external_inference_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"[{now()}] External inference complete: rows={len(output)} "
        f"folds={probabilities.shape[0]} output={output_path}"
    )
    return output


def train_one_ablation(
    args,
    mode: str,
    gate_spec: str,
    self_bundle: SelfTransformerEncoderBundle,
    encoder_mode: str,
    device: torch.device,
    root_outdir: str,
    use_gate_subdir: bool,
    split_frames: Optional[Dict[str, pd.DataFrame]] = None,
    split_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    set_seed(args.seed)
    gate_config = parse_gate_configuration(gate_spec, args.gate_init)
    if use_gate_subdir:
        mode_outdir = os.path.join(root_outdir, mode, gate_config.label)
    else:
        mode_outdir = os.path.join(root_outdir, mode)
    ensure_dir(mode_outdir)

    graph = build_graph_data(
        args,
        mode,
        device,
        split_frames=split_frames,
        split_metadata=split_metadata,
    )
    graph.metadata["gate_configuration"] = {
        "requested": gate_config.requested,
        "kind": gate_config.kind,
        "value": gate_config.value,
        "label": gate_config.label,
    }
    with open(os.path.join(mode_outdir, "graph_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(graph.metadata, handle, indent=2, ensure_ascii=False)

    model = SelfPretrainedGraphModel(
        args,
        graph,
        self_bundle,
        encoder_mode,
        gate_spec=gate_spec,
    ).to(device)
    model.set_encoder_trainability("frozen")
    optimizer = make_stage2_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience,
    )
    pos_weight = get_pos_weight(graph.train_pairs.label, args.pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = make_grad_scaler(args, device)

    best_score = -float("inf")
    best_epoch = -1
    wait = 0
    best_path = os.path.join(mode_outdir, "stage2_best.pt")
    if args.evaluate_only:
        best_path = resolve_existing_stage2_checkpoint(
            args=args,
            mode=mode,
            gate_config=gate_config,
            encoder_mode=encoder_mode,
            split_metadata=split_metadata,
        )
        print(
            f"[{now()}] Evaluation-only mode={mode} gate={gate_config.label}; "
            f"checkpoint={best_path}"
        )
    history: List[Dict[str, object]] = []

    print(
        f"[{now()}] Stage2 mode={mode} gate={gate_config.label} "
        f"pos_weight={float(pos_weight.item()):.4f} "
        f"freeze_encoder_epochs={args.freeze_encoder_epochs} "
        f"unfreeze_mode={args.unfreeze_mode} "
        f"gate_lr_multiplier={args.gate_lr_multiplier:g}"
    )

    n_epochs = 0 if args.evaluate_only else args.stage2_epochs
    for epoch in range(1, n_epochs + 1):
        if epoch == args.freeze_encoder_epochs + 1:
            model.set_encoder_trainability(args.unfreeze_mode)
            trainable = sum(p.numel() for p in model.sequence_parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.sequence_parameters())
            print(
                f"[{now()}] Stage2 mode={mode} gate={gate_config.label}: "
                f"encoder unfreeze {args.unfreeze_mode}; trainable={trainable}/{total}"
            )

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(args, device):
            node_embeddings = model.encode_graph(graph, args, apply_feature_mask=True)
            binding_logits = model.binding_logits(node_embeddings, graph.train_pairs)
            binding_loss = criterion(binding_logits, graph.train_pairs.label)

            vdj_loss = sample_auxiliary_loss(
                model, node_embeddings, graph, "vdj", args.aux_edge_batch_size,
            )
            state_loss = sample_auxiliary_loss(
                model, node_embeddings, graph, "state", args.aux_edge_batch_size,
            )
            total_loss = (
                binding_loss
                + args.lambda_vdj * vdj_loss
                + args.lambda_state * state_loss
            )

        scaler.scale(total_loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        val_loss, val_metrics, _ = evaluate_stage2(
            model,
            graph,
            graph.val_pairs,
            args,
            criterion,
        )
        score = val_metrics.get("aupr", float("nan"))
        if math.isnan(score):
            score = -val_loss
        scheduler.step(score)

        train_metrics = binary_metrics(
            graph.train_pairs.label.detach().cpu().numpy(),
            binding_logits.detach().cpu().numpy(),
        )
        row: Dict[str, object] = {
            "epoch": epoch,
            "mode": mode,
            "gate_requested": gate_config.requested,
            "gate_kind": gate_config.kind,
            "gate_label": gate_config.label,
            "gate_initial_or_fixed_value": gate_config.value,
            "total_loss": float(total_loss.detach().cpu().item()),
            "binding_loss": float(binding_loss.detach().cpu().item()),
            "vdj_aux_loss": float(vdj_loss.detach().cpu().item()),
            "state_aux_loss": float(state_loss.detach().cpu().item()),
            "val_loss": val_loss,
            "encoder_trainability": (
                "frozen" if epoch <= args.freeze_encoder_epochs else args.unfreeze_mode
            ),
            "graph_lr": optimizer_group_lr(optimizer, "graph"),
            "gate_lr": optimizer_group_lr(optimizer, "gate"),
            "encoder_lr": optimizer_group_lr(optimizer, "encoder"),
        }
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        row.update(model.gate_statistics())
        row.update(model.graph_diagnostics())
        history.append(row)

        print(
            f"[{now()}] Stage2 {mode} gate={gate_config.label} epoch={epoch:03d} "
            f"loss={row['total_loss']:.5f} bind={row['binding_loss']:.5f} "
            f"vdj={row['vdj_aux_loss']:.5f} state={row['state_aux_loss']:.5f} "
            f"gates(vdj/state)={row['vdj_gate_mean']:.3f}/{row['state_gate_mean']:.3f} "
            f"context_ratio={row['context_total_correction_norm_ratio']:.3f} "
            f"val {format_metrics(val_metrics)}"
        )

        if score > best_score + args.stage2_min_delta:
            best_score = score
            best_epoch = epoch
            wait = 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "mode": mode,
                "gate_configuration": {
                    "requested": gate_config.requested,
                    "kind": gate_config.kind,
                    "value": gate_config.value,
                    "label": gate_config.label,
                },
                "graph_metadata": graph.metadata,
                "args": vars(args),
                "self_tcr_checkpoint": self_bundle.tcr.checkpoint_path,
                "self_peptide_checkpoint": self_bundle.peptide.checkpoint_path,
                "encoder_mode": encoder_mode,
                "pretrained_tcr": bool(model.encoder_settings["pretrained_tcr"]),
                "pretrained_peptide": bool(model.encoder_settings["pretrained_peptide"]),
            }, best_path)
        else:
            wait += 1
            if args.stage2_early_stop > 0 and wait >= args.stage2_early_stop:
                print(
                    f"[{now()}] Stage2 {mode} gate={gate_config.label}: "
                    f"early stop at epoch={epoch}"
                )
                break

    if not args.evaluate_only:
        pd.DataFrame(history).to_csv(
            os.path.join(mode_outdir, "stage2_history.csv"),
            index=False,
        )
    checkpoint = load_stage2_checkpoint_compat(model, best_path, device)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Invalid Stage-2 checkpoint format: {best_path}")
    saved_mode = checkpoint.get("mode")
    if saved_mode is not None:
        saved_canonical_mode = mode_settings(str(saved_mode))["canonical_mode"]
        current_canonical_mode = mode_settings(mode)["canonical_mode"]
        if saved_canonical_mode != current_canonical_mode:
            raise ValueError(
                f"Checkpoint mode mismatch: checkpoint={saved_mode!r}, "
                f"requested={mode!r}."
            )
    saved_gate = checkpoint.get("gate_configuration")
    if isinstance(saved_gate, dict):
        saved_gate_label = saved_gate.get("label")
        if saved_gate_label and str(saved_gate_label) != gate_config.label:
            raise ValueError(
                f"Checkpoint gate mismatch: checkpoint={saved_gate_label!r}, "
                f"requested={gate_config.label!r}. Pass the same --gate_mode "
                "used during training."
            )
    if args.evaluate_only:
        best_epoch = int(checkpoint.get("epoch", -1))
        best_score = None

    split_results: Dict[str, Tuple[float, Dict[str, float], np.ndarray]] = {}
    for split_name, pairs in [
        ("train", graph.train_pairs),
        ("val", graph.val_pairs),
        ("test", graph.test_pairs),
    ]:
        split_results[split_name] = evaluate_stage2(
            model,
            graph,
            pairs,
            args,
            criterion,
        )
        if args.save_predictions:
            save_predictions(
                os.path.join(mode_outdir, f"{split_name}_predictions.csv"),
                pairs,
                split_results[split_name][2],
            )

    interpretability_summary: Dict[str, object] = {}
    requested_interpretability_mode = str(args.interpretability_mode).strip()
    target_interpretability_mode = mode_settings(
        requested_interpretability_mode
    )["canonical_mode"]
    if (
        (args.save_interpretability or args.save_transformer_attention)
        and graph.metadata["settings"]["canonical_mode"] == target_interpretability_mode
    ):
        interpretability_split = str(args.interpretability_split)
        if interpretability_split not in split_results:
            raise ValueError(
                f"Unknown interpretability split={interpretability_split}; "
                "expected train, val, or test."
            )
        interpretability_summary["graph_attention"] = export_graph_attention(
            model=model,
            graph=graph,
            args=args,
            mode_outdir=mode_outdir,
            split_name=interpretability_split,
        )
        if args.save_transformer_attention:
            interpretability_summary["transformer_attention"] = (
                export_transformer_attention(
                    model=model,
                    graph=graph,
                    args=args,
                    mode_outdir=mode_outdir,
                    split_name=interpretability_split,
                )
            )

    summary: Dict[str, object] = {
        "mode": mode,
        "graph_mode_canonical": graph.metadata["settings"]["canonical_mode"],
        "gate_requested": gate_config.requested,
        "gate_kind": gate_config.kind,
        "gate_label": gate_config.label,
        "gate_initial_or_fixed_value": gate_config.value,
        "gate_lr_multiplier": args.gate_lr_multiplier,
        "best_epoch": best_epoch,
        "best_val_selection_score": best_score,
        "encoder_mode": encoder_mode,
        "encoder_mode_canonical": model.encoder_settings["canonical_mode"],
        "self_tcr_checkpoint": self_bundle.tcr.checkpoint_path,
        "self_peptide_checkpoint": self_bundle.peptide.checkpoint_path,
        "self_tcr_max_len": self_bundle.tcr.max_len,
        "self_peptide_max_len": self_bundle.peptide.max_len,
        "self_tcr_hidden_dim": self_bundle.tcr.hidden_dim,
        "self_peptide_hidden_dim": self_bundle.peptide.hidden_dim,
        "self_tcr_layers": self_bundle.tcr.num_layers,
        "self_peptide_layers": self_bundle.peptide.num_layers,
        "self_n_heads_tcr": self_bundle.tcr.n_heads,
        "self_n_heads_peptide": self_bundle.peptide.n_heads,
        "pretrained_tcr": bool(model.encoder_settings["pretrained_tcr"]),
        "pretrained_peptide": bool(model.encoder_settings["pretrained_peptide"]),
        "labels_used_in_sequence_pretraining": False,
        "tcr_input": "cdr3b_only",
        "decoder_has_raw_sequence_shortcut": False,
        "tcr_instance_has_direct_sequence_encoding": False,
        "peptide_bypasses_context_graph": True,
        "reverse_hub_aggregation_enabled": False,
        "directed_context_hierarchy": (
            "TRBV/TRBJ->CDR3B_sequence->TCR_instance<-Tcellstate"
        ),
        "explicit_cdr3a_used": False,
        "alpha_vj_nodes_used": False,
        "beta_node_key": args.beta_node_key,
        "graph_dim": args.graph_dim,
        "use_sequence": bool(graph.metadata["settings"].get("use_sequence", True)),
        "use_graph": bool(
            graph.metadata["settings"].get(
                "use_graph",
                graph.metadata["settings"].get(
                    "learned_graph_message_passing", False
                ),
            )
        ),
        "final_model": "full_no_residue",
        "residue_graph_removed": True,
        "graph_attention_mode": model.graph_attention_mode,
        "transformer_attention_exposed": bool(args.save_transformer_attention),
        "context_integration": graph.metadata["settings"]["context_integration"],
        "learned_graph_message_passing": bool(
            graph.metadata["settings"]["learned_graph_message_passing"]
        ),
        "directed_context_steps": int(
            bool(graph.metadata["settings"]["learned_graph_message_passing"])
            * (
                int(bool(graph.metadata["settings"]["use_vdj"]))
                + int(bool(graph.metadata["settings"]["use_state"]))
            )
        ),
        "tcr_feature_mask_rate": args.tcr_feature_mask_rate,
        "peptide_feature_mask_rate": args.peptide_feature_mask_rate,
        "lambda_vdj": args.lambda_vdj,
        "lambda_state": args.lambda_state,
        "interpretability_enabled": bool(
            args.save_interpretability or args.save_transformer_attention
        ),
        "interpretability_mode": requested_interpretability_mode,
        "interpretability_split": str(args.interpretability_split),
    }
    if interpretability_summary:
        summary["interpretability_summary"] = interpretability_summary
    if split_metadata:
        for key, value in split_metadata.items():
            summary[f"split_{key}"] = value
    summary.update(model.gate_statistics())
    summary.update(model.graph_diagnostics())
    for split_name, (loss, metrics, _) in split_results.items():
        summary[f"{split_name}_loss"] = loss
        for key, value in metrics.items():
            summary[f"{split_name}_{key}"] = value

    with open(
        os.path.join(mode_outdir, "stage2_summary.json"),
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(
        f"[{now()}] Stage2 {mode} gate={gate_config.label} "
        f"best_epoch={best_epoch} test {format_metrics(split_results['test'][1])}"
    )
    return summary
