


from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "[PAD]"
MASK_TOKEN = "[MASK]"
CLS_TOKEN = "[CLS]"
UNK_TOKEN = "[UNK]"

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

TOKEN_TO_ID: Dict[str, int] = {
    PAD_TOKEN: 0,
    MASK_TOKEN: 1,
    CLS_TOKEN: 2,
    UNK_TOKEN: 3,
}
for aa in AMINO_ACIDS:
    TOKEN_TO_ID[aa] = len(TOKEN_TO_ID)

ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

PAD_ID = TOKEN_TO_ID[PAD_TOKEN]
MASK_ID = TOKEN_TO_ID[MASK_TOKEN]
CLS_ID = TOKEN_TO_ID[CLS_TOKEN]
UNK_ID = TOKEN_TO_ID[UNK_TOKEN]
AA_START_ID = TOKEN_TO_ID[AMINO_ACIDS[0]]
AA_END_ID = TOKEN_TO_ID[AMINO_ACIDS[-1]]
VOCAB_SIZE = len(TOKEN_TO_ID)



def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_torch_save(obj: object, path: Path) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def atomic_csv_write(rows: List[Dict[str, object]], path: Path) -> None:
    if not rows:
        raise ValueError("summary rows can not empty.")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(rows[0].keys())
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)



def read_clean_sequences(
    path: Path,
    min_length: int,
    max_length: int,
) -> Tuple[List[str], Dict[str, int]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    stats = {
        "raw_lines": 0,
        "empty_lines": 0,
        "invalid_character_lines": 0,
        "too_short_lines": 0,
        "too_long_lines": 0,
        "duplicate_lines": 0,
        "valid_unique_sequences": 0,
    }

    sequences: List[str] = []
    seen = set()
    allowed = set(AMINO_ACIDS)

    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            stats["raw_lines"] += 1
            seq = line.strip().upper()

            if not seq:
                stats["empty_lines"] += 1
                continue
            if len(seq) < min_length:
                stats["too_short_lines"] += 1
                continue
            if len(seq) > max_length:
                stats["too_long_lines"] += 1
                continue
            if any(char not in allowed for char in seq):
                stats["invalid_character_lines"] += 1
                continue
            if seq in seen:
                stats["duplicate_lines"] += 1
                continue

            seen.add(seq)
            sequences.append(seq)

    stats["valid_unique_sequences"] = len(sequences)

    if len(sequences) < 100:
        raise ValueError(
            f"{path} contains only {len(sequences)} valid unique sequences after cleaning; too few sequences."
        )

    return sequences, stats


def split_sequences(
    sequences: Sequence[str],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str], List[str]]:
    if val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("val_ratio and test_ratio must be greater than 0.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    n = len(sequences)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_val = max(1, int(round(n * val_ratio)))
    n_test = max(1, int(round(n * test_ratio)))
    n_train = n - n_val - n_test

    if n_train <= 0:
        raise ValueError("The training set is empty; adjust the split ratios.")

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train = [sequences[i] for i in train_idx]
    val = [sequences[i] for i in val_idx]
    test = [sequences[i] for i in test_idx]

    return train, val, test


class SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[str]) -> None:
        self.sequences = list(sequences)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> str:
        return self.sequences[index]


class MLMCollator:

    def __init__(
        self,
        mask_probability: float,
        evaluation_seed: Optional[int] = None,
    ) -> None:
        if not 0 < mask_probability < 1:
            raise ValueError("mask_probability must be in (0, 1).")

        self.mask_probability = mask_probability
        self.evaluation_seed = evaluation_seed
        self.generator: Optional[torch.Generator] = None
        self.reset()

    def reset(self) -> None:
        if self.evaluation_seed is None:
            self.generator = None
        else:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.evaluation_seed)

    def _rand(self, shape: torch.Size) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator)

    def _randint(
        self,
        low: int,
        high: int,
        shape: Tuple[int, ...],
    ) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=self.generator)

    def __call__(self, sequences: Sequence[str]) -> Dict[str, torch.Tensor]:
        encoded: List[List[int]] = []
        max_len = 0

        for seq in sequences:
            token_ids = [CLS_ID] + [TOKEN_TO_ID.get(aa, UNK_ID) for aa in seq]
            encoded.append(token_ids)
            max_len = max(max_len, len(token_ids))

        input_ids = torch.full(
            (len(encoded), max_len),
            fill_value=PAD_ID,
            dtype=torch.long,
        )

        for row, token_ids in enumerate(encoded):
            input_ids[row, : len(token_ids)] = torch.tensor(
                token_ids,
                dtype=torch.long,
            )

        attention_mask = input_ids.ne(PAD_ID)
        candidate_mask = attention_mask & input_ids.ne(CLS_ID)

        selected_mask = (
            self._rand(input_ids.shape) < self.mask_probability
        ) & candidate_mask

        no_mask_rows = torch.where(selected_mask.sum(dim=1) == 0)[0]
        for row in no_mask_rows.tolist():
            valid_positions = torch.where(candidate_mask[row])[0]
            choice_idx = self._randint(
                0,
                len(valid_positions),
                (1,),
            ).item()
            selected_mask[row, valid_positions[choice_idx]] = True

        labels = torch.full_like(input_ids, fill_value=-100)
        labels[selected_mask] = input_ids[selected_mask]

        corrupted_ids = input_ids.clone()
        replace_draw = self._rand(input_ids.shape)

        replace_with_mask = selected_mask & (replace_draw < 0.80)
        replace_with_random = (
            selected_mask
            & (replace_draw >= 0.80)
            & (replace_draw < 0.90)
        )

        corrupted_ids[replace_with_mask] = MASK_ID

        random_count = int(replace_with_random.sum().item())
        if random_count > 0:
            corrupted_ids[replace_with_random] = self._randint(
                AA_START_ID,
                AA_END_ID + 1,
                (random_count,),
            )

        return {
            "input_ids": corrupted_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


@dataclass
class EncoderConfig:
    vocab_size: int
    max_residue_length: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float

    @property
    def max_input_length(self) -> int:
        return self.max_residue_length + 1


class SmallTransformerEncoder(nn.Module):

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()

        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.config = config

        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=PAD_ID,
        )
        self.position_embedding = nn.Embedding(
            config.max_input_length,
            config.hidden_size,
        )
        self.input_layer_norm = nn.LayerNorm(config.hidden_size)
        self.input_dropout = nn.Dropout(config.dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.ffn_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_size),
            enable_nested_tensor=False,
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[PAD_ID].zero_()
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_input_length:
            raise ValueError(
                f"Input length {seq_len} exceeds the model maximum length "
                f"{self.config.max_input_length}."
            )

        position_ids = torch.arange(
            seq_len,
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, seq_len)

        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(position_ids)
        )
        hidden = self.input_layer_norm(hidden)
        hidden = self.input_dropout(hidden)

        hidden = self.transformer(
            hidden,
            src_key_padding_mask=~attention_mask.bool(),
        )
        return hidden

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.forward(input_ids, attention_mask)

        residue_mask = attention_mask.bool() & input_ids.ne(CLS_ID)
        residue_mask_f = residue_mask.unsqueeze(-1).to(hidden.dtype)

        summed = (hidden * residue_mask_f).sum(dim=1)
        denominator = residue_mask_f.sum(dim=1).clamp_min(1.0)
        return summed / denominator


class MLMPretrainingModel(nn.Module):
    def __init__(self, encoder: SmallTransformerEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        hidden_size = encoder.config.hidden_size

        self.mlm_dense = nn.Linear(hidden_size, hidden_size)
        self.mlm_activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(hidden_size)
        self.mlm_decoder = nn.Linear(
            hidden_size,
            encoder.config.vocab_size,
            bias=False,
        )
        self.mlm_bias = nn.Parameter(
            torch.zeros(encoder.config.vocab_size)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)
        hidden = self.mlm_dense(hidden)
        hidden = self.mlm_activation(hidden)
        hidden = self.mlm_layer_norm(hidden)
        return self.mlm_decoder(hidden) + self.mlm_bias


def create_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        lower_name = name.lower()
        if (
            parameter.ndim == 1
            or lower_name.endswith(".bias")
            or "layer_norm" in lower_name
            or ".norm" in lower_name
        ):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)

    parameter_groups = [
        {
            "params": decay_params,
            "weight_decay": weight_decay,
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        parameter_groups,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def create_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = int(round(total_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        progress = (current_step - warmup_steps) / (
            total_steps - warmup_steps
        )
        progress = min(max(progress, 0.0), 1.0)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )


def resolve_amp(
    device: torch.device,
    amp_mode: str,
) -> Tuple[bool, torch.dtype, bool]:
    if device.type != "cuda" or amp_mode == "off":
        return False, torch.float32, False

    if amp_mode == "bf16":
        return True, torch.bfloat16, False
    if amp_mode == "fp16":
        return True, torch.float16, True

    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return True, torch.bfloat16, False
    return True, torch.float16, True


@torch.no_grad()
def evaluate_mlm(
    model: MLMPretrainingModel,
    loader: DataLoader,
    collator: MLMCollator,
    device: torch.device,
    autocast_enabled: bool,
    autocast_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    collator.reset()

    total_loss_sum = 0.0
    total_correct = 0
    total_masked = 0

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum",
    )

    for batch in loader:
        input_ids = batch["input_ids"].to(
            device,
            non_blocking=True,
        )
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        labels = batch["labels"].to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_enabled,
        ):
            logits = model(input_ids, attention_mask)
            loss_sum = criterion(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
            )

        masked = labels.ne(-100)
        predictions = logits.argmax(dim=-1)

        total_loss_sum += float(loss_sum.item())
        total_correct += int(
            (predictions[masked] == labels[masked]).sum().item()
        )
        total_masked += int(masked.sum().item())

    if total_masked == 0:
        raise RuntimeError("No masked tokens were found in the evaluation data.")

    mean_loss = total_loss_sum / total_masked
    accuracy = total_correct / total_masked
    perplexity = math.exp(min(mean_loss, 20.0))

    return {
        "loss": mean_loss,
        "masked_accuracy": accuracy,
        "perplexity": perplexity,
        "masked_tokens": float(total_masked),
    }


def train_one_encoder(
    modality: str,
    input_path: Path,
    output_path: Path,
    min_length: int,
    max_length: int,
    mask_probability: float,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float,
    epochs: int,
    batch_size: int,
    eval_batch_size: int,
    grad_accum_steps: int,
    learning_rate: float,
    weight_decay: float,
    warmup_ratio: float,
    min_lr: float,
    gradient_clip: float,
    patience: int,
    min_delta: float,
    val_ratio: float,
    test_ratio: float,
    num_workers: int,
    device: torch.device,
    amp_mode: str,
    seed: int,
) -> Dict[str, object]:
    set_seed(seed)

    log(f"{modality}: Reading and cleaning {input_path}")
    sequences, clean_stats = read_clean_sequences(
        input_path,
        min_length=min_length,
        max_length=max_length,
    )

    train_sequences, val_sequences, test_sequences = split_sequences(
        sequences,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    log(
        f"{modality}: valid_unique={len(sequences):,}, "
        f"train={len(train_sequences):,}, "
        f"val={len(val_sequences):,}, "
        f"test={len(test_sequences):,}"
    )

    train_dataset = SequenceDataset(train_sequences)
    val_dataset = SequenceDataset(val_sequences)
    test_dataset = SequenceDataset(test_sequences)

    train_collator = MLMCollator(
        mask_probability=mask_probability,
        evaluation_seed=None,
    )
    val_collator = MLMCollator(
        mask_probability=mask_probability,
        evaluation_seed=seed + 101,
    )
    test_collator = MLMCollator(
        mask_probability=mask_probability,
        evaluation_seed=seed + 202,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader_kwargs = dict(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collator,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=loader_generator,
        drop_last=False,
    )
    if num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=val_collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=test_collator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    config = EncoderConfig(
        vocab_size=VOCAB_SIZE,
        max_residue_length=max_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )

    encoder = SmallTransformerEncoder(config)
    model = MLMPretrainingModel(encoder).to(device)

    encoder_params = sum(
        parameter.numel()
        for parameter in model.encoder.parameters()
    )
    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    optimizer = create_optimizer(
        model,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    updates_per_epoch = math.ceil(
        len(train_loader) / grad_accum_steps
    )
    total_update_steps = max(1, updates_per_epoch * epochs)
    min_lr_ratio = min_lr / learning_rate

    scheduler = create_cosine_scheduler(
        optimizer,
        total_steps=total_update_steps,
        warmup_ratio=warmup_ratio,
        min_lr_ratio=min_lr_ratio,
    )

    (
        autocast_enabled,
        autocast_dtype,
        grad_scaler_enabled,
    ) = resolve_amp(device, amp_mode)

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            scaler = torch.amp.GradScaler(
                "cuda",
                enabled=grad_scaler_enabled,
            )
        except TypeError:
            scaler = torch.amp.GradScaler(
                enabled=grad_scaler_enabled,
            )
    else:
        scaler = torch.cuda.amp.GradScaler(
            enabled=grad_scaler_enabled
        )

    criterion = nn.CrossEntropyLoss(
        ignore_index=-100,
        reduction="sum",
    )

    best_val_loss = float("inf")
    best_val_metrics: Dict[str, float] = {}
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0
    epochs_ran = 0
    start_time = time.time()

    log(
        f"{modality}: model hidden={hidden_size}, layers={num_layers}, "
        f"heads={num_heads}, ffn={ffn_dim}, "
        f"encoder_params={encoder_params:,}, total_pretrain_params={total_params:,}"
    )
    log(
        f"{modality}: batch={batch_size}, grad_accum={grad_accum_steps}, "
        f"effective_batch={batch_size * grad_accum_steps}, "
        f"epochs={epochs}, lr={learning_rate:g}, mask_prob={mask_probability}"
    )

    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)

        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_masked = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            input_ids = batch["input_ids"].to(
                device,
                non_blocking=True,
            )
            attention_mask = batch["attention_mask"].to(
                device,
                non_blocking=True,
            )
            labels = batch["labels"].to(
                device,
                non_blocking=True,
            )

            masked_count = int(labels.ne(-100).sum().item())

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(input_ids, attention_mask)
                loss_sum = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
                mean_loss = loss_sum / max(masked_count, 1)
                backward_loss = mean_loss / grad_accum_steps

            scaler.scale(backward_loss).backward()

            with torch.no_grad():
                masked = labels.ne(-100)
                predictions = logits.argmax(dim=-1)
                epoch_correct += int(
                    (
                        predictions[masked]
                        == labels[masked]
                    ).sum().item()
                )
                epoch_masked += masked_count
                epoch_loss_sum += float(loss_sum.item())

            should_update = (
                batch_index % grad_accum_steps == 0
                or batch_index == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        train_loss = epoch_loss_sum / max(epoch_masked, 1)
        train_accuracy = epoch_correct / max(epoch_masked, 1)

        val_metrics = evaluate_mlm(
            model=model,
            loader=val_loader,
            collator=val_collator,
            device=device,
            autocast_enabled=autocast_enabled,
            autocast_dtype=autocast_dtype,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        log(
            f"{modality} epoch={epoch:03d} "
            f"train_loss={train_loss:.6f} "
            f"train_acc={train_accuracy:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_acc={val_metrics['masked_accuracy']:.4f} "
            f"val_ppl={val_metrics['perplexity']:.4f} "
            f"lr={current_lr:.3e}"
        )

        improved = (
            val_metrics["loss"]
            < best_val_loss - min_delta
        )
        if improved:
            best_val_loss = val_metrics["loss"]
            best_val_metrics = dict(val_metrics)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            log(
                f"{modality}: early stop at epoch={epoch}; "
                f"best_epoch={best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(f"{modality}: No valid checkpoint was obtained.")

    model.load_state_dict(best_state)
    model.to(device)

    test_metrics = evaluate_mlm(
        model=model,
        loader=test_loader,
        collator=test_collator,
        device=device,
        autocast_enabled=autocast_enabled,
        autocast_dtype=autocast_dtype,
    )

    elapsed_seconds = time.time() - start_time

    encoder_state = {
        key: value.detach().cpu()
        for key, value in model.encoder.state_dict().items()
    }

    checkpoint = {
        "format_version": 1,
        "model_type": "small_transformer_encoder",
        "modality": modality,
        "pretraining_objective": "masked_language_modeling",
        "pooling": "mean_over_residues_excluding_cls_and_pad",
        "config": asdict(config),
        "special_token_ids": {
            "pad_id": PAD_ID,
            "mask_id": MASK_ID,
            "cls_id": CLS_ID,
            "unk_id": UNK_ID,
        },
        "token_to_id": TOKEN_TO_ID,
        "id_to_token": ID_TO_TOKEN,
        "state_dict": encoder_state,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_metrics["loss"],
        "best_validation_masked_accuracy": best_val_metrics[
            "masked_accuracy"
        ],
        "test_loss": test_metrics["loss"],
        "test_masked_accuracy": test_metrics[
            "masked_accuracy"
        ],
        "seed": seed,
    }

    atomic_torch_save(checkpoint, output_path)

    log(
        f"{modality}: saved encoder={output_path}; "
        f"best_epoch={best_epoch}, "
        f"val_loss={best_val_metrics['loss']:.6f}, "
        f"test_loss={test_metrics['loss']:.6f}, "
        f"test_acc={test_metrics['masked_accuracy']:.4f}"
    )

    summary_row: Dict[str, object] = {
        "modality": modality,
        "input_file": str(input_path),
        "encoder_file": str(output_path),
        "raw_lines": clean_stats["raw_lines"],
        "empty_lines": clean_stats["empty_lines"],
        "invalid_character_lines": clean_stats[
            "invalid_character_lines"
        ],
        "too_short_lines": clean_stats["too_short_lines"],
        "too_long_lines": clean_stats["too_long_lines"],
        "duplicate_lines": clean_stats["duplicate_lines"],
        "valid_unique_sequences": clean_stats[
            "valid_unique_sequences"
        ],
        "train_sequences": len(train_sequences),
        "validation_sequences": len(val_sequences),
        "test_sequences": len(test_sequences),
        "min_residue_length": min_length,
        "max_residue_length": max_length,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "ffn_dim": ffn_dim,
        "dropout": dropout,
        "mask_probability": mask_probability,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "effective_batch_size": (
            batch_size * grad_accum_steps
        ),
        "learning_rate": learning_rate,
        "minimum_learning_rate": min_lr,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "gradient_clip": gradient_clip,
        "max_epochs": epochs,
        "epochs_ran": epochs_ran,
        "early_stop_patience": patience,
        "best_epoch": best_epoch,
        "best_validation_loss": round(
            best_val_metrics["loss"],
            8,
        ),
        "best_validation_masked_accuracy": round(
            best_val_metrics["masked_accuracy"],
            8,
        ),
        "best_validation_perplexity": round(
            best_val_metrics["perplexity"],
            8,
        ),
        "test_loss": round(test_metrics["loss"], 8),
        "test_masked_accuracy": round(
            test_metrics["masked_accuracy"],
            8,
        ),
        "test_perplexity": round(
            test_metrics["perplexity"],
            8,
        ),
        "encoder_parameters": encoder_params,
        "pretraining_total_parameters": total_params,
        "training_seconds": round(elapsed_seconds, 2),
        "seed": seed,
        "amp_mode": amp_mode,
        "device": str(device),
    }

    del model
    del optimizer
    del scheduler
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary_row


def load_frozen_encoder(
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Tuple[SmallTransformerEncoder, Dict[str, object]]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    config = EncoderConfig(**checkpoint["config"])
    encoder = SmallTransformerEncoder(config)
    encoder.load_state_dict(
        checkpoint["state_dict"],
        strict=True,
    )
    encoder.to(device)
    encoder.eval()

    for parameter in encoder.parameters():
        parameter.requires_grad = False

    return encoder, checkpoint


def build_parser() -> argparse.ArgumentParser:
    corpus_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Train compact MLM Transformer encoders for "
            "TCR and peptide sequences."
        )
    )

    parser.add_argument(
        "--tcr_file",
        type=str,
        default=str(corpus_dir / "tcr.txt"),
        help="Path to the unlabeled TCR corpus, one sequence per line.",
    )
    parser.add_argument(
        "--peptide_file",
        type=str,
        default=str(corpus_dir / "epitope.txt"),
        help="Path to the unlabeled peptide corpus, one sequence per line.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="results/pretrain",
        help="Output directory for the two encoder checkpoints and summary.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--amp",
        choices=["auto", "fp16", "bf16", "off"],
        default="auto",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--min_learning_rate",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--gradient_clip",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--ffn_dim",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--tcr_min_length",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--tcr_max_length",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--tcr_mask_probability",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--tcr_epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--peptide_min_length",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--peptide_max_length",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--peptide_mask_probability",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--peptide_epochs",
        type=int,
        default=25,
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.batch_size <= 0:
        raise ValueError("batch_size must be greater than 0.")
    if args.grad_accum_steps <= 0:
        raise ValueError(
            "grad_accum_steps must be greater than 0."
        )
    if args.min_learning_rate > args.learning_rate:
        raise ValueError(
            "min_learning_rate must not exceed learning_rate."
        )

    if args.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable in the current environment."
            )
        device = torch.device(args.device)
    else:
        device = torch.device(args.device)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    tcr_output = outdir / "self_tcr_transformer_encoder.pt"
    peptide_output = (
        outdir / "self_peptide_transformer_encoder.pt"
    )
    summary_output = outdir / "pretrain_summary.csv"

    log(f"Device={device}")
    log(f"Output directory={outdir}")
    log(
        "Only three files will be written: "
        f"{tcr_output.name}, "
        f"{peptide_output.name}, "
        f"{summary_output.name}"
    )

    tcr_summary = train_one_encoder(
        modality="tcr",
        input_path=Path(args.tcr_file),
        output_path=tcr_output,
        min_length=args.tcr_min_length,
        max_length=args.tcr_max_length,
        mask_probability=args.tcr_mask_probability,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        epochs=args.tcr_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_learning_rate,
        gradient_clip=args.gradient_clip,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        device=device,
        amp_mode=args.amp,
        seed=args.seed,
    )

    peptide_summary = train_one_encoder(
        modality="peptide",
        input_path=Path(args.peptide_file),
        output_path=peptide_output,
        min_length=args.peptide_min_length,
        max_length=args.peptide_max_length,
        mask_probability=args.peptide_mask_probability,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        epochs=args.peptide_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        min_lr=args.min_learning_rate,
        gradient_clip=args.gradient_clip,
        patience=args.early_stop_patience,
        min_delta=args.early_stop_min_delta,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        device=device,
        amp_mode=args.amp,
        seed=args.seed + 1,
    )

    atomic_csv_write(
        [tcr_summary, peptide_summary],
        summary_output,
    )

    log(f"Saved summary={summary_output}")
    log("Training completed.")


if __name__ == "__main__":
    main()
