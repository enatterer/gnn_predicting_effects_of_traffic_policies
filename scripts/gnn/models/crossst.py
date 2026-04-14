import os
import sys
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from torch_geometric.nn import TransformerConv, GraphNorm

# Add 'scripts' dir to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN
from gnn.help_functions import validate_model_during_training, select_target_tensor, LinearWarmupCosineDecayScheduler


class _PatternEncoder(nn.Module):
    """
    CrossST-inspired encoder block with:
    - Temporal pattern bank retrieval.
    - Spatial message passing + pattern bank retrieval.
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int, temporal_bank_size: int, spatial_bank_size: int, use_dropout: bool, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)

        self.temporal_bank = nn.Parameter(torch.randn(temporal_bank_size, hidden_dim) * 0.02)
        self.temporal_query = nn.Linear(hidden_dim, hidden_dim)

        self.spatial_conv = TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout if use_dropout else 0.0)
        self.spatial_norm = GraphNorm(hidden_dim)
        self.spatial_bank = nn.Parameter(torch.randn(spatial_bank_size, hidden_dim) * 0.02)
        self.spatial_query = nn.Linear(hidden_dim, hidden_dim)

        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(dropout) if use_dropout else nn.Identity()

    def _retrieve(self, x: torch.Tensor, query_layer: nn.Linear, bank: torch.Tensor) -> torch.Tensor:
        q = query_layer(x)
        logits = torch.matmul(q, bank.t())
        weights = torch.softmax(logits, dim=-1)
        return torch.matmul(weights, bank)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        h0 = self.in_proj(x)

        # Temporal-style bank retrieval (CrossST-inspired replacement of FFT bank)
        t_bank = self._retrieve(h0, self.temporal_query, self.temporal_bank)
        h_t = h0 + t_bank
        h_t = F.gelu(h_t)

        # Spatial-style message passing + bank retrieval
        h_s = self.spatial_conv(h_t, edge_index)
        h_s = self.spatial_norm(h_s)
        s_bank = self._retrieve(h_s, self.spatial_query, self.spatial_bank)
        h_s = h_s + s_bank
        h_s = F.gelu(h_s)

        h = self.fuse(torch.cat([h_t, h_s], dim=-1))
        h = self.dropout(F.gelu(h))
        return {"h_t": h_t, "h_s": h_s, "h": h}


class CrossST(BaseGNN):
    """
    CrossST-inspired adaptation for graph regression in this repository:
    - Pre-training mode: universal branch prediction only.
    - Fine-tuning mode: frozen universal + trainable personalized branch with
      KL + InfoNCE distillation.
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
        universal_dim: int = 128,
        personalized_dim: int = 64,
        temporal_bank_size: int = 32,
        spatial_bank_size: int = 32,
        num_heads: int = 4,
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

        self.universal_dim = universal_dim
        self.personalized_dim = personalized_dim
        self.temporal_bank_size = temporal_bank_size
        self.spatial_bank_size = spatial_bank_size
        self.num_heads = num_heads

        self.crossst_finetune_mode = False
        self.crossst_alpha = 0.3
        self.crossst_beta = 0.3
        self._last_outputs = {}

        self.define_layers()

    def define_layers(self):
        self.universal_encoder = _PatternEncoder(
            in_dim=self.in_channels,
            hidden_dim=self.universal_dim,
            num_heads=self.num_heads,
            temporal_bank_size=self.temporal_bank_size,
            spatial_bank_size=self.spatial_bank_size,
            use_dropout=self.use_dropout,
            dropout=self.dropout,
        )
        self.universal_head = nn.Linear(self.universal_dim, self.out_channels)

        self.personalized_encoder = _PatternEncoder(
            in_dim=self.in_channels,
            hidden_dim=self.personalized_dim,
            num_heads=self.num_heads,
            temporal_bank_size=max(8, self.temporal_bank_size // 2),
            spatial_bank_size=max(8, self.spatial_bank_size // 2),
            use_dropout=self.use_dropout,
            dropout=self.dropout,
        )

        self.t_proj = nn.Linear(self.personalized_dim, self.universal_dim)
        self.s_proj = nn.Linear(self.personalized_dim, self.universal_dim)
        self.t_filter = nn.Linear(self.universal_dim, self.universal_dim)
        self.s_filter = nn.Linear(self.universal_dim, self.universal_dim)
        self.finetune_head = nn.Linear(self.universal_dim * 2, self.out_channels)

    def enable_finetune_mode(self, alpha: float = 0.3, beta: float = 0.3):
        self.crossst_finetune_mode = True
        self.crossst_alpha = float(alpha)
        self.crossst_beta = float(beta)
        for p in self.universal_encoder.parameters():
            p.requires_grad = False
        for p in self.universal_head.parameters():
            p.requires_grad = False

    def disable_finetune_mode(self):
        self.crossst_finetune_mode = False
        for p in self.universal_encoder.parameters():
            p.requires_grad = True
        for p in self.universal_head.parameters():
            p.requires_grad = True

    def _cache(self, **kwargs):
        self._last_outputs = kwargs

    def forward(self, data):
        x = data.x.to(self.dtype)
        edge_index = data.edge_index

        if not self.crossst_finetune_mode:
            u = self.universal_encoder(x, edge_index)
            pred = self.universal_head(u["h"])
            self._cache(pred=pred, h_u_t=u["h_t"], h_u_s=u["h_s"])
            return pred.reshape(-1, 1)

        with torch.no_grad():
            u = self.universal_encoder(x, edge_index)

        p = self.personalized_encoder(x, edge_index)
        p_t = self.t_proj(p["h_t"])
        p_s = self.s_proj(p["h_s"])

        f_t = torch.sigmoid(self.t_filter(u["h_t"]))
        f_s = torch.sigmoid(self.s_filter(u["h_s"]))
        u_t_f = u["h_t"] * f_t
        u_s_f = u["h_s"] * f_s

        merged_t = p_t + u_t_f
        merged_s = p_s + u_s_f
        pred = self.finetune_head(torch.cat([merged_t, merged_s], dim=-1))

        self._cache(
            pred=pred,
            h_u_t=u["h_t"],
            h_u_s=u["h_s"],
            h_p_t=p_t,
            h_p_s=p_s,
        )
        return pred.reshape(-1, 1)

    @staticmethod
    def _distill_kl(h_u: torch.Tensor, h_p: torch.Tensor) -> torch.Tensor:
        p_u = F.softmax(h_u, dim=-1)
        log_p = F.log_softmax(h_p, dim=-1)
        return F.kl_div(log_p, p_u, reduction="batchmean")

    @staticmethod
    def _distill_infonce(h_u: torch.Tensor, h_p: torch.Tensor, temperature: float = 0.3) -> torch.Tensor:
        h_u_n = F.normalize(h_u, dim=-1)
        h_p_n = F.normalize(h_p, dim=-1)
        logits = torch.matmul(h_p_n, h_u_n.t()) / max(temperature, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def train_model(self, config=None, loss_fct=None, optimizer=None, train_dl=None, valid_dl=None, device=None, early_stopping=None, model_save_path=None, apply_source_city_weights=False, source_city_weights=None, city_weight_callback=None):
        if not self.crossst_finetune_mode:
            return super().train_model(
                config=config,
                loss_fct=loss_fct,
                optimizer=optimizer,
                train_dl=train_dl,
                valid_dl=valid_dl,
                device=device,
                early_stopping=early_stopping,
                model_save_path=model_save_path,
                apply_source_city_weights=apply_source_city_weights,
                source_city_weights=source_city_weights,
                city_weight_callback=city_weight_callback,
            )

        if config is None:
            raise ValueError("Config cannot be None")

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

        from training.help_functions import setup_wandb_metrics
        setup_wandb_metrics()

        for epoch in range(config.num_epochs):
            super().train()
            optimizer.zero_grad()
            epoch_train_loss = 0.0
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
                    predicted = self(data)
                    l_pred = loss_fct(predicted, targets, data, data.batch)
                    l_kl = self._distill_kl(self._last_outputs["h_u_t"], self._last_outputs["h_p_t"])
                    l_nce = self._distill_infonce(self._last_outputs["h_u_s"], self._last_outputs["h_p_s"])
                    train_loss = l_pred + self.crossst_alpha * l_kl + self.crossst_beta * l_nce

                epoch_train_loss += train_loss.item()
                scaler.scale(train_loss).backward()
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                wandb.log(
                    {
                        "batch_train_loss": train_loss.item(),
                        "batch_pred_loss": l_pred.item(),
                        "batch_kl_loss": l_kl.item(),
                        "batch_infonce_loss": l_nce.item(),
                        "batch_step": step,
                    }
                )

            if len(train_dl) % config.gradient_accumulation_steps != 0:
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
                "train_loss": epoch_train_loss / len(train_dl),
                "lr": epoch_start_lr,
                "r^2": r_squared,
                "spearman": spearman_corr,
                "pearson": pearson_corr,
                "epoch": epoch,
                "crossst_alpha": self.crossst_alpha,
                "crossst_beta": self.crossst_beta,
            }
            for key, value in hit_rates.items():
                log_dict[key] = value
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
                    },
                    checkpoint_path,
                )

            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()
        return best_val_loss, epoch
