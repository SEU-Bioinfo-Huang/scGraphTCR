# Encoder Pretraining

Run from the repository root. Each input file contains one amino-acid sequence per line. The study used TCRβ sequences from the Emerson repertoire and peptides from the Anthem dataset through the TEINet pretraining bundle. `tcr.txt` and `epitope.txt` are the inputs before the script's filtering and deduplication. To use other corpora, edit `--tcr_file` and `--peptide_file`. New outputs are saved in `results/pretrain`.

```bash
python data/pretrain/train_self_transformer_encoders.py \
  --tcr_file data/pretrain/tcr.txt \
  --peptide_file data/pretrain/epitope.txt \
  --outdir results/pretrain \
  --device cuda \
  --hidden_size 128 \
  --num_layers 2 \
  --num_heads 4 \
  --ffn_dim 512 \
  --dropout 0.10 \
  --batch_size 512 \
  --eval_batch_size 1024 \
  --grad_accum_steps 2 \
  --learning_rate 3e-4 \
  --min_learning_rate 1e-6 \
  --weight_decay 0.01 \
  --warmup_ratio 0.05 \
  --gradient_clip 1.0 \
  --tcr_mask_probability 0.15 \
  --peptide_mask_probability 0.20 \
  --tcr_epochs 20 \
  --peptide_epochs 25 \
  --early_stop_patience 4 \
  --seed 42 \
  --amp auto \
  --num_workers 4
```

The command trains both encoders and saves `self_tcr_transformer_encoder.pt` and `self_peptide_transformer_encoder.pt`. To use the new checkpoints downstream, pass their paths through `--self_tcr_encoder_path` and `--self_peptide_encoder_path`.
