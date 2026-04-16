import os
import sys
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

from torch_geometric.nn import GCNConv, GraphNorm

# Add 'scripts' dir to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN
from gnn.help_functions import (
    LinearWarmupCosineDecayScheduler,
    select_target_tensor,
    validate_model_during_training,
)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float):
        ctx.lambda_ = float(lambda_)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradientReversalFn.apply(x, lambda_)


class CityTrans(BaseGNN):
    """
    CityTrans-inspired domain-adversarial surrogate adapted to this repo's graph regression setting.

    Core CityTrans components implemented:
    - Feature extractor (graph encoder)
    - Self-adaptive ST-Knowledge (two banks: source/target)
    - Feature domain discriminator + knowledge domain discriminator (with gradient reversal)
    - Knowledge attention retrieval from ST-Knowledge
    - End-to-end single-stage training objective

    IMPORTANT:
    - This repository's data is not spatio-temporal sequences. We therefore implement the
      CityTrans training objective and modules on per-graph node features (policy-conditioned),
      while preserving the CityTrans adversarial + knowledge-attention structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        dropout: float = 0.1,
        use_dropout: bool = False,
        dtype: torch.dtype = torch.float32,
        log_to_wandb: bool = False,
        use_target_standardization: bool = False,
        target_normalization: str = None,
        hidden_dim: int = 128,
        num_gcn_layers: int = 2,
        knowledge_gk: int = 16,
        knowledge_pk: int = 12,
        knowledge_dk: int = 16,
        adv_lambda: float = 0.5,
        grl_lambda: float = 1.0,
        target_city: Optional[str] = None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            use_dropout=use_dropout,
            dtype=dtype,
            log_to_wandb=log_to_wandb,
            use_target_standardization=use_target_standardization,
            target_normalization=target_normalization,
        )

        self.hidden_dim = int(hidden_dim)
        self.num_gcn_layers = max(int(num_gcn_layers), 1)
        self.knowledge_gk = int(knowledge_gk)
        self.knowledge_pk = int(knowledge_pk)
        self.knowledge_dk = int(knowledge_dk)
        self.adv_lambda = float(adv_lambda)
        self.grl_lambda = float(grl_lambda)
        self.target_city = str(target_city) if target_city is not None else None

        self._last_losses: Dict[str, torch.Tensor] = {}

        self.define_layers()
        self.initialize_weights()

    def define_layers(self):
        self.in_proj = nn.Linear(self.in_channels, self.hidden_dim)

        self.gcn_layers = nn.ModuleList()
        self.gcn_norms = nn.ModuleList()
        for _ in range(self.num_gcn_layers):
            self.gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            self.gcn_norms.append(GraphNorm(self.hidden_dim))

        # ST-Knowledge banks: source and target (learnable parameters)
        # K ∈ R[Gk * Pk, dk]
        k_rows = self.knowledge_gk * self.knowledge_pk
        self.K_source = nn.Parameter(torch.randn(k_rows, self.knowledge_dk) * 0.02)
        self.K_target = nn.Parameter(torch.randn(k_rows, self.knowledge_dk) * 0.02)

        # Query projection from hidden representation to dk
        self.query_fc = nn.Linear(self.hidden_dim, self.knowledge_dk)
        # Predictor maps [h, z] -> y_hat
        self.pred_head = nn.Sequential(
            nn.Linear(self.hidden_dim + self.knowledge_dk, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.use_dropout else nn.Identity(),
            nn.Linear(self.hidden_dim, self.out_channels),
        )

        # Feature domain discriminator D_f: hidden -> 2 classes
        self.feature_discriminator = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )

        # Knowledge domain discriminator D_k: knowledge -> 2 classes
        self.knowledge_discriminator = nn.Sequential(
            nn.Linear(self.knowledge_dk, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def _domain_labels_per_node(self, data) -> torch.Tensor:
        """
        Build per-node binary domain labels:
        - 0: source domain (city != target_city)
        - 1: target domain (city == target_city)
        """
        if self.target_city is None:
            raise ValueError("CityTrans requires `target_city` to assign domain labels.")
        if not hasattr(data, "city"):
            raise ValueError("Batch is missing `data.city`; cannot assign domain labels.")
        if not hasattr(data, "batch") or data.batch is None:
            # Single graph: all nodes belong to the same city
            city = str(data.city)
            label = 1 if city == self.target_city else 0
            return torch.full((data.x.size(0),), label, device=data.x.device, dtype=torch.long)

        # Batched: data.city is a list of length num_graphs (PyG Batch behavior)
        num_graphs = int(data.batch.max().item() + 1)
        cities = data.city if isinstance(data.city, list) else [data.city] * num_graphs
        if len(cities) != num_graphs:
            # Defensive: fall back to repeating the single city string if provided
            cities = [cities[0]] * num_graphs

        graph_labels = torch.tensor(
            [1 if str(c) == self.target_city else 0 for c in cities],
            device=data.x.device,
            dtype=torch.long,
        )
        return graph_labels[data.batch]

    def _knowledge_attention(self, h: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Knowledge attention module (CityTrans Eq. 10-12):
        - q = FC(h) in R[dk]
        - weights = softmax(q K^T)
        - z = weights K
        """
        q = self.query_fc(h)  # [N, dk]
        logits = torch.matmul(q, K.t())  # [N, k_rows]
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, K)  # [N, dk]

    def forward(self, data):
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        h = self.in_proj(x)
        h = F.relu(h)

        for conv, norm in zip(self.gcn_layers, self.gcn_norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
            if self.use_dropout:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Split nodes into source vs target for using respective knowledge banks
        domain_y = self._domain_labels_per_node(data)  # [N]
        z = torch.zeros((h.size(0), self.knowledge_dk), device=h.device, dtype=h.dtype)

        src_mask = domain_y == 0
        tgt_mask = domain_y == 1
        if src_mask.any():
            z[src_mask] = self._knowledge_attention(h[src_mask], self.K_source).to(z.dtype)
        if tgt_mask.any():
            z[tgt_mask] = self._knowledge_attention(h[tgt_mask], self.K_target).to(z.dtype)

        pred = self.pred_head(torch.cat([h, z], dim=-1))
        return pred.reshape(-1, 1)

    def _compute_adversarial_losses(self, data, h_detached_or_not: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute feature and knowledge adversarial losses with gradient reversal.
        """
        domain_y = self._domain_labels_per_node(data)  # [N]

        # Feature discriminator operates on per-node representations.
        # If the batch has only one domain (e.g., source-only training/validation),
        # feature adversarial loss is ill-posed; skip it.
        if int(domain_y.unique().numel()) < 2:
            lf_adv = h_detached_or_not.new_zeros(())
        else:
            feat_in = grad_reverse(h_detached_or_not, self.grl_lambda)
            feat_logits = self.feature_discriminator(feat_in)  # [N, 2]
            lf_adv = F.cross_entropy(feat_logits, domain_y)

        # Knowledge discriminator: classify K_source as source (0), K_target as target (1).
        k_src = grad_reverse(self.K_source, self.grl_lambda)
        k_tgt = grad_reverse(self.K_target, self.grl_lambda)
        k_logits_src = self.knowledge_discriminator(k_src)  # [k_rows, 2]
        k_logits_tgt = self.knowledge_discriminator(k_tgt)  # [k_rows, 2]
        y_src = torch.zeros((k_logits_src.size(0),), device=k_logits_src.device, dtype=torch.long)
        y_tgt = torch.ones((k_logits_tgt.size(0),), device=k_logits_tgt.device, dtype=torch.long)
        lk_adv = 0.5 * (F.cross_entropy(k_logits_src, y_src) + F.cross_entropy(k_logits_tgt, y_tgt))

        return lf_adv, lk_adv

    def train_model(
        self,
        config=None,
        loss_fct=None,
        optimizer=None,
        train_dl=None,
        valid_dl=None,
        device=None,
        early_stopping=None,
        model_save_path=None,
        apply_source_city_weights: bool = False,
        source_city_weights: dict = None,
        city_weight_callback=None,
    ):
        if config is None:
            raise ValueError("Config cannot be None")
        if self.target_city is None:
            raise ValueError("CityTrans requires `target_city` to be set (model_kwargs).")

        if self.target_normalization == "relative_standard_scaler":
            self.compute_target_statistics(train_dl, model_save_path.replace(".pt", "_target_stats.pt"), config, device)

        scaler = GradScaler()
        total_steps = config.num_epochs * len(train_dl)
        scheduler = LinearWarmupCosineDecayScheduler(
            initial_lr=config.initial_lr,
            total_steps=total_steps,
            peak_lr=config.peak_lr,
            warmup_fraction=config.warmup_fraction,
            min_lr_fraction=config.min_lr_fraction,
            cosine_decay_rate=config.cosine_decay_rate,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = scheduler.initial_lr

        best_val_loss = float("inf")
        checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        do_wandb = bool(wandb is not None and getattr(wandb, "run", None) is not None)
        if do_wandb:
            from training.help_functions import setup_wandb_metrics

            setup_wandb_metrics()

        active_source_city_weights = dict(source_city_weights or {})

        for epoch in range(config.num_epochs):
            if city_weight_callback is not None:
                callback_weights = city_weight_callback(epoch=epoch, model=self)
                if callback_weights:
                    active_source_city_weights = dict(callback_weights)

            super().train()
            optimizer.zero_grad()
            epoch_train_loss = 0.0
            epoch_train_pred_loss = 0.0
            epoch_train_adv_f = 0.0
            epoch_train_adv_k = 0.0

            epoch_start_lr = scheduler.get_lr(epoch * len(train_dl))

            for idx, data in tqdm(enumerate(train_dl), total=len(train_dl), desc=f"Epoch {epoch+1}/{config.num_epochs}"):
                step = epoch * len(train_dl) + idx
                lr = scheduler.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                data = data.to(device)
                targets = select_target_tensor(data, config.target_type)
                targets = self.standardize_target(targets, data=data)

                with autocast():
                    # Forward + prediction loss
                    pred = self(data)
                    l_pred = loss_fct(pred, targets, data, data.batch)
                    if apply_source_city_weights:
                        city_weight = self._compute_batch_city_weight(data, active_source_city_weights)
                        l_pred = l_pred * city_weight

                    # Adversarial losses (feature+knowledge) per CityTrans
                    # We need the internal node representation; recompute it deterministically here.
                    x = data.x.to(self.dtype)
                    h = F.relu(self.in_proj(x))
                    for conv, norm in zip(self.gcn_layers, self.gcn_norms):
                        h = conv(h, data.edge_index)
                        h = norm(h)
                        h = F.relu(h)
                        if self.use_dropout:
                            h = F.dropout(h, p=self.dropout, training=self.training)

                    lf_adv, lk_adv = self._compute_adversarial_losses(data=data, h_detached_or_not=h)
                    train_loss = l_pred + self.adv_lambda * (lf_adv + lk_adv)

                epoch_train_loss += float(train_loss.item())
                epoch_train_pred_loss += float(l_pred.item())
                epoch_train_adv_f += float(lf_adv.item())
                epoch_train_adv_k += float(lk_adv.item())

                scaler.scale(train_loss).backward()
                if config.use_gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if do_wandb:
                    wandb.log(
                        {
                            # Match standard repo logging keys (BaseGNN/CrossST).
                            "batch_train_loss": float(train_loss.item()),
                            "batch_step": step,
                        }
                    )

            if len(train_dl) % config.gradient_accumulation_steps != 0:
                if config.use_gradient_clipping:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            val_result = validate_model_during_training(
                config=config,
                model=self,
                dataset=valid_dl,
                loss_func=loss_fct,
                device=device,
            )
            if len(val_result) == 5:
                val_loss, r_squared, spearman_corr, pearson_corr, hit_rates = val_result
            else:
                val_loss, r_squared, spearman_corr, pearson_corr = val_result
                hit_rates = {}

            log_dict = {
                "val_loss": val_loss,
                "train_loss": epoch_train_loss / max(len(train_dl), 1),
                "lr": epoch_start_lr,
                "r^2": r_squared,
                "spearman": spearman_corr,
                "pearson": pearson_corr,
                "epoch": epoch,
            }
            for key, value in hit_rates.items():
                log_dict[key] = value
            if do_wandb:
                wandb.log(log_dict)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_save_path:
                    torch.save(self.state_dict(), model_save_path)

            if epoch % 20 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "val_loss": val_loss,
                        "target_normalization": self.target_normalization,
                        "citytrans_target_city": self.target_city,
                        "citytrans_adv_lambda": self.adv_lambda,
                        "citytrans_grl_lambda": self.grl_lambda,
                    },
                    checkpoint_path,
                )

            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

        if do_wandb:
            wandb.summary["best_val_loss"] = best_val_loss
            wandb.finish()
        return best_val_loss, epoch

