import argparse
import copy
import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from transformers import GPT2Config
from sklearn.decomposition import PCA
from tqdm import tqdm

from src.sbtg.data.transformer_variants import create_transformer_variant

class TransformerDataset(torch.utils.data.Dataset):
    def __init__(self, seqs, masks, labels=None):
        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.seqs[idx], self.masks[idx], self.labels[idx]
        return self.seqs[idx], self.masks[idx], -1

def calculate_loss(logits, targets, masks):
    """
    Calculates cross entropy loss only on answer-bearing positions where mask=True.
    Shift targets and masks since predicting next token.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = targets[..., 1:].contiguous()
    shift_masks  = masks[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size())

    # Calculate mask mean
    loss = (loss * shift_masks.float()).sum() / (shift_masks.float().sum() + 1e-8)
    return loss

def train_model(model, train_loader, val_loaders, device, epochs, lr=1e-3, early_stopping_patience=3):
    """
    Train the model and return a history dict suitable for parity plots (F1).

    Returns
    -------
    history : dict with key "epochs" — a list of per-epoch records containing
        train_loss, per-family val_losses, and mean_val_loss.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.to(device)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {"epochs": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        total_batches = 0

        for seqs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            seqs  = seqs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=seqs)

            loss = calculate_loss(outputs.logits, seqs, masks)
            loss.backward()
            optimizer.step()

            train_loss   += loss.item()
            total_batches += 1

        avg_train_loss = train_loss / total_batches
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        total_val_loss = 0
        val_metrics = {}

        with torch.no_grad():
            for fam_name, v_loader in val_loaders.items():
                fam_loss = 0
                for seqs, masks, _ in v_loader:
                    seqs  = seqs.to(device)
                    masks = masks.to(device)
                    outputs = model(input_ids=seqs)
                    loss = calculate_loss(outputs.logits, seqs, masks)
                    fam_loss += loss.item()
                avg_fam_loss = fam_loss / max(1, len(v_loader))
                val_metrics[fam_name] = float(avg_fam_loss)
                total_val_loss += avg_fam_loss
                print(f"  > Val {fam_name}: {avg_fam_loss:.4f}")

        avg_val_loss = total_val_loss / len(val_loaders)
        print(f"  > Mean Val Loss: {avg_val_loss:.4f}")

        history["epochs"].append({
            "epoch":          epoch + 1,
            "train_loss":     float(avg_train_loss),
            "val_losses":     val_metrics,
            "mean_val_loss":  float(avg_val_loss),
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered due to plateauing validation loss.")
                break

    return history

def extract_activations(model, data_loader, device):
    model.eval()
    all_hidden_states = []

    with torch.no_grad():
        for seqs, masks, _ in data_loader:
            seqs = seqs.to(device)
            outputs = model(input_ids=seqs, output_hidden_states=True)
            # outputs.hidden_states: tuple of (batch, seq_len, hidden_size), length = num_layers + 1
            batch_hidden = torch.stack(outputs.hidden_states, dim=1).cpu().numpy()
            all_hidden_states.append(batch_hidden)

    return np.concatenate(all_hidden_states, axis=0)


def extract_logits_pca(model, data_loader, device, pca_dim=256):
    """Extract output logits and reduce with PCA.

    Returns
    -------
    reduced : ndarray, shape (N, seq_len, pca_dim)
    pca     : fitted sklearn PCA object (needed for Phase 2 projection)
    """
    model.eval()
    all_logits = []
    with torch.no_grad():
        for seqs, masks, _ in data_loader:
            seqs = seqs.to(device)
            outputs = model(input_ids=seqs)
            all_logits.append(outputs.logits.cpu().numpy())
    raw = np.concatenate(all_logits, axis=0)  # (N, seq_len, vocab_size)
    N, S, V = raw.shape
    flat = raw.reshape(-1, V)
    effective_dim = min(pca_dim, V, flat.shape[0])
    if effective_dim < pca_dim:
        print(f"    Logit PCA: clamping n_components from {pca_dim} to {effective_dim} "
              f"(vocab_size={V})")
    pca = PCA(n_components=effective_dim)
    reduced = pca.fit_transform(flat).reshape(N, S, effective_dim)
    print(f"    Logit PCA: {V} → {effective_dim}  "
          f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")
    return reduced, pca


def compute_attention_statistics(model, data_loader, device, n_layers, n_heads, seq_len):
    """
    Compute per-layer/head attention statistics aggregated over the data loader.
    Saves:
      mean_attended_distance  (n_layers, n_heads)
      mean_entropy            (n_layers, n_heads)
      recency_k5_fraction     (n_layers, n_heads)  fraction of mass on ≤5-step-away keys
    Returns dict of numpy arrays.
    """
    model.eval()
    stats = {
        "mean_attended_distance": np.zeros((n_layers, n_heads)),
        "mean_entropy":           np.zeros((n_layers, n_heads)),
        "recency_k5_fraction":    np.zeros((n_layers, n_heads)),
    }
    n_batches = 0

    # Pre-compute distance matrix (seq_len × seq_len)
    i_pos = np.arange(seq_len).reshape(-1, 1)
    j_pos = np.arange(seq_len).reshape(1, -1)
    dist_mat = np.abs(i_pos - j_pos).astype(np.float32)      # (seq_len, seq_len)
    recency_mask = (dist_mat <= 5).astype(np.float32)         # (seq_len, seq_len)

    # NOTE: HF >=4.36 uses a separate GPT2SdpaAttention class when
    # `_attn_implementation="sdpa"`, which is the default.  SDPA never returns
    # attention weights, and setting `_attn_implementation="eager"` on an
    # already-constructed instance is a no-op because the forward method is
    # bound at construction.  Caller must pass a model that was built in eager
    # mode; we defensively assert and warn.
    for block in model.transformer.h:
        inner = getattr(block.attn, "old_attn", block.attn)
        impl = getattr(inner, "_attn_implementation", "eager")
        if impl != "eager":
            print(f"  WARNING: GPT2Attention._attn_implementation='{impl}' — "
                  f"attention weights will NOT be returned. Rebuild the model "
                  f"with config._attn_implementation='eager' before this call.")

    with torch.no_grad():
        for seqs, masks, _ in data_loader:
            seqs = seqs.to(device)
            outputs = model(input_ids=seqs, output_attentions=True)

            if outputs.attentions is None:
                print("  WARNING: outputs.attentions is None — "
                      "attention statistics will be all-zero.")
                break

            for l_idx, attn_l in enumerate(outputs.attentions):
                # attn_l: (batch, n_heads, seq_q, seq_k)
                attn_np = attn_l.cpu().numpy()  # (B, H, S, S)

                mean_dist = np.mean(
                    np.sum(attn_np * dist_mat[None, None, :, :], axis=-1), axis=(0, 2)
                )
                stats["mean_attended_distance"][l_idx] += mean_dist

                eps = 1e-10
                entropy = np.mean(
                    -np.sum(attn_np * np.log(attn_np + eps), axis=-1), axis=(0, 2)
                )
                stats["mean_entropy"][l_idx] += entropy

                recency_frac = np.mean(
                    np.sum(attn_np * recency_mask[None, None, :, :], axis=-1), axis=(0, 2)
                )
                stats["recency_k5_fraction"][l_idx] += recency_frac

            n_batches += 1

    if n_batches > 0:
        for k in stats:
            stats[k] /= n_batches
    else:
        print("  WARNING: 0 batches processed — attention statistics are all-zero.")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   type=str, required=True)
    parser.add_argument("--out-dir",    type=str, required=True)
    parser.add_argument("--epochs",     type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device",     type=str,
                        default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--pe-types",   nargs="+", default=["rope", "alibi", "absolute"])
    parser.add_argument("--save-logits", action="store_true",
                        help="Extract and save PCA-reduced logits for logit-space analysis")
    parser.add_argument("--logit-pca-dim", type=int, default=256,
                        help="PCA dimensions for logit reduction (default: 256)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training; load existing model.pt and only extract logits/acts")
    parser.add_argument("--rope-base", type=float, default=10000.0,
                        help="RoPE base θ (default 10000, standard RoFormer). "
                             "Only applies when 'rope' is in --pe-types.  Used by the "
                             "RoPE base sweep experiment (cluster_rope_base_sweep.sh).")
    # Architecture overrides — defaults match the main paper's matched-model
    # config (4L × 256H × 4H × MLP 1024).  For the small base × context sweep
    # (cluster_rope_base_context_grid.sh) we use 2L × 128H × 2H × MLP 512 so
    # 30+ training runs fit in one cluster job.
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-embd",  type=int, default=256)
    parser.add_argument("--n-head",  type=int, default=4)
    parser.add_argument("--n-inner", type=int, default=1024)

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "metadata.json", "r") as f:
        meta = json.load(f)

    families = meta.get("families", [
        "variable_lag_copy", "absolute_anchor", "order_sensitive",
        "distance_bucket", "iid_random",
    ])

    config = GPT2Config(
        vocab_size=meta["vocab_size"],
        n_positions=meta["seq_len"],
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_inner=args.n_inner,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )

    # Load all data upfront
    train_data = []
    val_data   = {}
    test_data  = {}

    for f in families:
        t_s = np.load(data_dir / f"{f}_train.npy")
        t_m = np.load(data_dir / f"{f}_train_mask.npy")

        v_s = np.load(data_dir / f"{f}_val.npy")
        v_m = np.load(data_dir / f"{f}_val_mask.npy")

        ts_s = np.load(data_dir / f"{f}_test.npy")
        ts_m = np.load(data_dir / f"{f}_test_mask.npy")

        f_idx = families.index(f)
        train_data.append((t_s, t_m, np.full(len(t_s), f_idx)))
        val_data[f]  = TransformerDataset(v_s,  v_m,  np.full(len(v_s),  f_idx))
        test_data[f] = TransformerDataset(ts_s, ts_m, np.full(len(ts_s), f_idx))

    # Pool train data
    pooled_t_s      = np.concatenate([d[0] for d in train_data], axis=0)
    pooled_t_m      = np.concatenate([d[1] for d in train_data], axis=0)
    pooled_labels   = np.concatenate([d[2] for d in train_data], axis=0)

    pooled_train_dataset = TransformerDataset(pooled_t_s, pooled_t_m, pooled_labels)
    pooled_test_dataset  = TransformerDataset(
        np.concatenate([d.seqs.numpy()  for d in test_data.values()], axis=0),
        np.concatenate([d.masks.numpy() for d in test_data.values()], axis=0),
        np.concatenate([d.labels        for d in test_data.values()], axis=0),
    )

    for pe_type in args.pe_types:
        seed = args.seed
        print(f"\n=============================================")
        print(f"Training {pe_type.upper()} seed {seed}...")
        print(f"=============================================")
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = create_transformer_variant(config, pe_type, rope_base=args.rope_base)
        if pe_type == "rope" and args.rope_base != 10000.0:
            print(f"  RoPE base = {args.rope_base:g} (non-standard)")

        train_loader = torch.utils.data.DataLoader(
            pooled_train_dataset, batch_size=args.batch_size, shuffle=True
        )
        val_loaders  = {
            f: torch.utils.data.DataLoader(d, batch_size=args.batch_size)
            for f, d in val_data.items()
        }
        test_loader = torch.utils.data.DataLoader(
            pooled_test_dataset, batch_size=args.batch_size
        )

        save_dir = out_dir / f"{pe_type}_seed{seed}"
        save_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_training:
            # Load existing weights — skip training entirely
            ckpt = save_dir / "model.pt"
            if not ckpt.exists():
                print(f"  ERROR: --skip-training but {ckpt} not found, skipping {pe_type}")
                continue
            model.to(args.device)
            model.load_state_dict(torch.load(ckpt, map_location=args.device))
            print(f"  Loaded existing model from {ckpt}")
            history = None
        else:
            history = train_model(model, train_loader, val_loaders, args.device, args.epochs)

        # Extract raw activations (PCA is computed in the analysis script)
        test_acts  = extract_activations(model, test_loader,  args.device)
        train_acts = extract_activations(model, train_loader, args.device)

        np.save(save_dir / "train_acts.npy",         train_acts)
        np.save(save_dir / "test_acts.npy",          test_acts)
        np.save(save_dir / "test_family_labels.npy", pooled_test_dataset.labels)

        # --- Logit extraction (optional) ---
        if args.save_logits:
            print(f"  Extracting PCA-reduced logits (dim={args.logit_pca_dim}) ...")
            # Fit PCA on train logits, apply to both train and test
            train_logits_pca, logit_pca_obj = extract_logits_pca(
                model, train_loader, args.device, pca_dim=args.logit_pca_dim)
            # For test: extract raw logits, then project with train-fitted PCA
            model.eval()
            test_logits_raw = []
            with torch.no_grad():
                for seqs, masks, _ in test_loader:
                    seqs = seqs.to(args.device)
                    outputs = model(input_ids=seqs)
                    test_logits_raw.append(outputs.logits.cpu().numpy())
            test_raw = np.concatenate(test_logits_raw, axis=0)
            N_t, S_t, V_t = test_raw.shape
            actual_pca_dim = logit_pca_obj.n_components_
            test_logits_pca = logit_pca_obj.transform(
                test_raw.reshape(-1, V_t)).reshape(N_t, S_t, actual_pca_dim)
            np.save(save_dir / "train_logits_pca.npy", train_logits_pca)
            np.save(save_dir / "test_logits_pca.npy",  test_logits_pca)
            with open(save_dir / "logit_pca.pkl", "wb") as fp:
                pickle.dump(logit_pca_obj, fp)
            print(f"    Saved logit arrays to {save_dir}")

        # Compute and save aggregated attention statistics.
        #
        # HF builds GPT2 with SDPA attention by default; SDPA never returns
        # attention weights.  Rebuild an eager-mode clone of the same model,
        # load the trained weights into it, and run the stats pass on the
        # eager clone.  This keeps training fast while still exposing weights.
        if not (save_dir / "attn_stats.json").exists():
            print(f"  Computing attention statistics (rebuilding eager-mode clone)...")
            eager_config = copy.deepcopy(config)
            eager_config._attn_implementation = "eager"
            eager_model = create_transformer_variant(eager_config, pe_type, rope_base=args.rope_base)
            eager_model.load_state_dict(model.state_dict())
            eager_model.to(args.device)
            eager_model.eval()

            n_layers = config.n_layer
            n_heads  = config.n_head
            seq_len  = meta["seq_len"]
            attn_stats = compute_attention_statistics(
                eager_model, test_loader, args.device, n_layers, n_heads, seq_len
            )
            del eager_model
            attn_stats_serialisable = {k: v.tolist() for k, v in attn_stats.items()}
            with open(save_dir / "attn_stats.json", "w") as f:
                json.dump(attn_stats_serialisable, f, indent=2)
        else:
            print(f"  Attention stats already exist — skipping")

        # Save per-epoch training history for parity plots (F1)
        if history is not None:
            history["pe_type"] = pe_type
            history["seed"]    = seed
            with open(save_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)

            torch.save(model.state_dict(), save_dir / "model.pt")
        print(f"  Saved to {save_dir}")

if __name__ == "__main__":
    main()
