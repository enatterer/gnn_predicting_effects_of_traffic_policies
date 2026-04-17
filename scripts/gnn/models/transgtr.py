import os
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, GraphNorm

# Add 'scripts' dir to Python Path
scripts_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from gnn.models.base_gnn import BaseGNN


class TransGTR(BaseGNN):
    """
    TransGTR-inspired model adapted to this repository's graph setup.

    Core ideas mirrored from the paper:
    - Node feature network (city-agnostic representation via distillation-like loss).
    - Structure generator that predicts transferable edge strengths.
    - Forecasting model trained on generated structures.

    In strict source-only pretraining mode, we use:
    - task loss + source self-distillation auxiliary loss.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        hidden_dim: int = 128,
        structure_hidden_dim: int = 64,
        dropout: float = 0.1,
        use_dropout: bool = False,
        dtype: torch.dtype = torch.float32,
        log_to_wandb: bool = False,
        use_target_standardization: bool = False,
        target_normalization: str = None,
        lambda_distill: float = 0.1,
        lambda_structure_smooth: float = 0.01,
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
        self.hidden_dim = hidden_dim
        self.structure_hidden_dim = structure_hidden_dim
        self.lambda_distill = float(lambda_distill)
        self.lambda_structure_smooth = float(lambda_structure_smooth)
        self._last_aux = {}
        self.define_layers()
        self.initialize_weights()

    def define_layers(self):
        # Node feature network
        self.node_encoder = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout if self.use_dropout else 0.0),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Structure generator over existing edges
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.structure_hidden_dim),
            nn.GELU(),
            nn.Linear(self.structure_hidden_dim, 1),
        )

        # Forecasting model on generated structure
        self.conv1 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.norm1 = GraphNorm(self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.norm2 = GraphNorm(self.hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(self.hidden_dim // 2, self.out_channels),
        )

    def _generate_edge_weight(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)
        logits = self.edge_mlp(edge_feat).squeeze(-1)
        return torch.sigmoid(logits)

    def _encode_two_views(self, x: torch.Tensor):
        # Teacher view (stable)
        h_teacher = self.node_encoder(x)
        # Student view (noisy) for source-only distillation
        noise = 0.05 * torch.randn_like(x)
        h_student = self.node_encoder(x + noise)
        return h_teacher, h_student

    def forward(self, data):
        x, edge_index = data.x.to(self.dtype), data.edge_index

        h_teacher, h_student = self._encode_two_views(x)
        edge_weight = self._generate_edge_weight(h_teacher, edge_index)

        h = self.conv1(h_teacher, edge_index, edge_weight=edge_weight)
        h = self.norm1(F.gelu(h))
        if self.use_dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = self.norm2(F.gelu(h))
        if self.use_dropout:
            h = F.dropout(h, p=self.dropout, training=self.training)

        out = self.head(h).reshape(-1, 1)

        # Store auxiliaries used by train_model
        self._last_aux = {
            "h_teacher": h_teacher,
            "h_student": h_student,
            "edge_weight": edge_weight,
        }
        return out

    def compute_auxiliary_loss(self) -> torch.Tensor:
        h_teacher = self._last_aux["h_teacher"].detach()
        h_student = self._last_aux["h_student"]
        edge_weight = self._last_aux["edge_weight"]

        # Distillation-like node feature alignment
        l_distill = F.mse_loss(h_student, h_teacher)
        # Encourage confident/sparse structure (paper-inspired regularization spirit)
        l_struct = (edge_weight * (1.0 - edge_weight)).mean()
        return self.lambda_distill * l_distill + self.lambda_structure_smooth * l_struct

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
        apply_source_city_weights=False,
        source_city_weights=None,
        city_weight_callback=None,
    ):
        # Reuse base training, but include TransGTR auxiliary losses by monkey-patching
        if config is None:
            raise ValueError("Config cannot be None")

        # Compute target stats when requested by base behavior.
        if self.target_normalization == "relative_standard_scaler":
            self.compute_target_statistics(train_dl, model_save_path.replace(".pt", "_target_stats.pt"), config, device)

        from gnn.help_functions import LinearWarmupCosineDecayScheduler, validate_model_during_training, select_target_tensor
        from training.help_functions import setup_wandb_metrics
        import wandb

        # Stability-first: run TransGTR pretraining/fine-tuning in full fp32.
        total_steps = config.num_epochs * len(train_dl)
        scheduler = LinearWarmupCosineDecayScheduler(
            initial_lr=config.initial_lr,
            total_steps=total_steps,
            peak_lr=config.peak_lr,
            warmup_fraction=config.warmup_fraction,
            min_lr_fraction=config.min_lr_fraction,
            cosine_decay_rate=config.cosine_decay_rate,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = scheduler.initial_lr

        best_val_loss = float("inf")
        checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
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
            epoch_start_lr = scheduler.get_lr(epoch * len(train_dl))

            for idx, data in enumerate(train_dl):
                step = epoch * len(train_dl) + idx
                lr = scheduler.get_lr(step)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                data = data.to(device)
                targets = select_target_tensor(data, config.target_type)
                targets = self.standardize_target(targets, data=data)

                pred = self(data)
                pred_loss = loss_fct(pred, targets, data, data.batch)
                aux_loss = self.compute_auxiliary_loss()
                train_loss = pred_loss + aux_loss

                if apply_source_city_weights:
                    city_weight = self._compute_batch_city_weight(data, active_source_city_weights)
                    train_loss = train_loss * city_weight

                if not torch.isfinite(pred).all():
                    raise RuntimeError(f"Non-finite prediction detected in TransGTR at epoch={epoch}, batch={idx}")
                if not torch.isfinite(train_loss):
                    raise RuntimeError(
                        f"Non-finite loss detected in TransGTR at epoch={epoch}, batch={idx}, "
                        f"pred_loss={float(pred_loss.detach().cpu())}, aux_loss={float(aux_loss.detach().cpu())}"
                    )

                epoch_train_loss += train_loss.item()
                train_loss.backward()

                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    if config.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                wandb.log(
                    {
                        "batch_train_loss": train_loss.item(),
                        "batch_pred_loss": pred_loss.item(),
                        "batch_aux_loss": aux_loss.item(),
                        "batch_step": step,
                    }
                )

            if len(train_dl) % config.gradient_accumulation_steps != 0:
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            val_result = validate_model_during_training(
                config=config,
                model=self,
                dataset=valid_dl,
                loss_func=loss_fct,
                device=device,
            )
            if len(val_result) == 5:
                val_loss, r2, sp, pr, hit_rates = val_result
            else:
                val_loss, r2, sp, pr = val_result
                hit_rates = {}

            log_dict = {
                "val_loss": val_loss,
                "train_loss": epoch_train_loss / len(train_dl),
                "lr": epoch_start_lr,
                "r^2": r2,
                "spearman": sp,
                "pearson": pr,
                "epoch": epoch,
            }
            for k, v in hit_rates.items():
                log_dict[k] = v
            wandb.log(log_dict)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_save_path:
                    torch.save(self.state_dict(), model_save_path)

            if epoch % 20 == 0:
                ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "val_loss": val_loss,
                    },
                    ckpt_path,
                )

            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()
        return best_val_loss, epoch
