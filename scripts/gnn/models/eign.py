"""
This file implements the architecture from the paper:
"EIGN: Efficient and Interpretable Graph Neural Networks" (https://arxiv.org/abs/2410.16935)
The implementation can be found here: https://github.com/dfuchsgruber/eign/tree/main
As all models in this repository, this model is a Graph Neural Network (GNN) that predicts the effects of traffic policies.

The parameters UseMonteCarloDropout and PredictModeStats may be implemented in the future.
"""

import os
import sys
from abc import ABC, abstractmethod

from tqdm import tqdm
import wandb
import numpy as np
from .base_gnn import BaseGNN
from abc import abstractmethod
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .block import EIGNBlock, EIGNBlockMagneticEdgeLaplacianConv

from gnn.help_functions import (
    validate_model_during_training_eign,
    LinearWarmupCosineDecayScheduler,
)


class EIGNOutput(NamedTuple):
    signed: torch.Tensor | None
    unsigned: torch.Tensor | None


class EIGN(BaseGNN):
    def __init__(
        self,
        in_channels_signed: int | None,
        out_channels_signed: int | None,
        hidden_channels_signed: int,
        in_channels_unsigned: int,
        hidden_channels_unsigned: int,
        out_channels_unsigned: int,
        num_blocks: int,
        dropout: float = 0.1,
        use_dropout: bool = False,
        predict_mode_stats: bool = False,
        dtype: torch.dtype = torch.float32,
        verbose: bool = False,
        signed_activation_fn=F.tanh,
        unsigned_activation_fn=F.relu,
        **kwargs_block,
    ):
        super().__init__(
            in_channels=1,
            out_channels=1,
            dropout=dropout,
            use_dropout=use_dropout,
            predict_mode_stats=predict_mode_stats,
            dtype=dtype,
            verbose=verbose,
        )

        self.out_channels_signed = out_channels_signed
        self.out_channels_unsigned = out_channels_unsigned
        self.signed_activation_fn = signed_activation_fn
        self.unsigned_activation_fn = unsigned_activation_fn

        self.dropout = nn.Dropout(p=dropout)

        self.blocks = nn.ModuleList()
        _in_channels_signed = in_channels_signed
        _in_channels_unsigned = in_channels_unsigned

        for _ in range(num_blocks):
            block = self.initialize_block(
                in_channels_signed=_in_channels_signed,  # type: ignore
                out_channels_signed=hidden_channels_signed,
                in_channels_unsigned=_in_channels_unsigned,
                out_channels_unsigned=hidden_channels_unsigned,
                signed_activation_fn=signed_activation_fn,
                unsigned_activation_fn=unsigned_activation_fn,
                **kwargs_block,
            )
            self.blocks.append(block)
            _in_channels_signed = hidden_channels_signed
            _in_channels_unsigned = hidden_channels_unsigned

        if self.out_channels_signed:
            self.signed_head = nn.Linear(
                hidden_channels_signed, out_channels_signed, bias=False
            )
        else:
            self.register_buffer("signed_head", None)
        
        if self.out_channels_unsigned:
            self.unsigned_head = nn.Linear(
                hidden_channels_unsigned, out_channels_unsigned, bias=False
            )
        else:
            self.register_buffer("unsigned_head", None)

    @abstractmethod
    def initialize_block(
        self,
        in_channels_signed: int,
        out_channels_signed: int,
        in_channels_unsigned: int,
        out_channels_unsigned: int,
        signed_activation_fn=F.tanh,
        unsigned_activation_fn=F.relu,
        *args,
        **kwargs,
    ) -> EIGNBlock:
        raise NotImplementedError

    def forward(
        self,
        x_signed: torch.Tensor | None,
        x_unsigned: torch.Tensor | None,
        edge_index: torch.Tensor,
        is_directed: torch.Tensor,
        *args,
        **kwargs,
    ) -> EIGNOutput:

        # change x_signed to float32
        if x_signed is not None:
            x_signed = x_signed.to(torch.float32)
        if x_unsigned is not None:
            x_unsigned = x_unsigned.to(torch.float32)
        
        print(f"enter forward")
        for block in self.blocks:
            x_signed, x_unsigned = block(
                x_signed=x_signed,
                x_unsigned=x_unsigned,
                edge_index=edge_index,
                is_directed=is_directed,
            )

            if x_signed is not None:
                x_signed = self.signed_activation_fn(x_signed)
                x_signed = self.dropout(x_signed)
            if x_unsigned is not None:
                x_unsigned = self.unsigned_activation_fn(x_unsigned)
                x_unsigned = self.dropout(x_unsigned)
        print(f"out of forward")

        if self.out_channels_signed:
            x_signed = self.signed_head(x_signed)
        else:
            x_signed = None
        if self.out_channels_unsigned:
            x_unsigned = self.unsigned_head(x_unsigned)
        else:
            x_unsigned = None

        return EIGNOutput(signed=x_signed, unsigned=x_unsigned)

    def train_model(
        self,
        config: object = None,
        loss_fct: nn.Module = None,
        optimizer: optim.Optimizer = None,
        train_dl: DataLoader = None,
        valid_dl: DataLoader = None,
        device: torch.device = None,
        early_stopping: object = None,
        model_save_path: str = None,
        scalers_train: dict = None,
        scalers_validation: dict = None,
    ) -> tuple:
        """
        Overriding train_model method from base_gnn
        """
        if config is None:
            raise ValueError("Config cannot be None")

        scaler = GradScaler()
        total_steps = config.num_epochs * len(train_dl)
        scheduler = LinearWarmupCosineDecayScheduler(
            initial_lr=config.lr, total_steps=total_steps
        )
        best_val_loss = float("inf")
        checkpoint_dir = os.path.join(os.path.dirname(model_save_path), "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(config.num_epochs):
            super().train()
            optimizer.zero_grad()
            for idx, data in tqdm(
                enumerate(train_dl),
                total=len(train_dl),
                desc=f"Epoch {epoch+1}/{config.num_epochs}",
            ):
                step = epoch * len(train_dl) + idx
                lr = scheduler.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                # Ensure data is in float32
                data = data.to(device)
                targets_node_predictions_signed = data.y_signed.to(torch.float32)
                targets_node_predictions_unsigned = data.y.to(torch.float32)

                # x_unscaled = scalers_train["x_scaler"].inverse_transform(
                #     data.x.detach().clone().cpu().numpy()
                # )

                if config.predict_mode_stats:
                    raise NotImplementedError(
                        "Predicting mode stats is not implemented yet."
                    )
                    # targets_mode_stats = data.mode_stats

                with autocast():
                    # Forward pass
                    if config.predict_mode_stats:
                        raise NotImplementedError(
                            "Predicting mode stats is not implemented yet."
                        )
                        # predicted, mode_stats_pred = self(
                        #     x_signed=data.x_signed,
                        #     x_unsigned=data.x,
                        #     edge_index=data.edge_index,
                        #     is_directed=data.edge_is_directed,
                        # )
                        # train_loss_node_predictions = loss_fct(
                        #     predicted, targets_node_predictions, x_unscaled
                        # )
                        # train_loss_mode_stats = loss_fct(
                        #     mode_stats_pred, targets_mode_stats
                        # )  # add weight here also later!
                        # train_loss = train_loss_node_predictions + train_loss_mode_stats
                    else:
                        # Ensure inputs to model are float32
                        x_signed = data.x_signed.to(torch.float32)
                        x_unsigned = data.x.to(torch.float32)
                        
                        print(f"epoch: {epoch}")
                        print(f"eign x_signed: {x_signed.shape}")
                        print(f"eign x_unsigned: {x_unsigned.shape}\n")
                        
                        eign_output = self(
                            x_signed=x_signed,
                            x_unsigned=x_unsigned,
                            edge_index=data.edge_index,
                            is_directed=data.edge_is_directed,
                        )
                        predicted_signed, predicted_unsigned = eign_output.signed, eign_output.unsigned
                        
                        # Ensure predictions and targets are float32
                        predicted_signed = predicted_signed.to(torch.float32)
                        predicted_unsigned = predicted_unsigned.to(torch.float32)
                        
                        train_loss = (
                            loss_fct(
                                predicted_signed,
                                targets_node_predictions_signed,
                            )
                            + loss_fct(
                                predicted_unsigned,
                                targets_node_predictions_unsigned,
                            )
                        )

                # Backward pass
                train_loss = train_loss.to(dtype=torch.float32)
                scaler.scale(train_loss).backward()

                # Gradient clipping
                if config.use_gradient_clipping:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                if (idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                # Do not log train loss at every iteration, as it uses CPU
                if (idx + 1) % 10 == 0:
                    if config.predict_mode_stats:
                        raise NotImplementedError(
                            "Predicting mode stats is not implemented yet."
                        )
                        # wandb.log(
                        #     {
                        #         "train_loss": train_loss.item(),
                        #         "epoch": epoch,
                        #         "train_loss_node_predictions_signed": train_loss_node_predictions_signed.item(),
                        #         "train_loss_node_predictions_unsigned": train_loss_node_predictions_signed.item(),
                                
                        #         "train_loss_mode_stats": train_loss_mode_stats.item(),
                        #     }
                        # )
                    else:
                        wandb.log({"train_loss": train_loss.item(), "epoch": epoch})

            if len(train_dl) % config.gradient_accumulation_steps != 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Validation step
            if config.predict_mode_stats:
                raise NotImplementedError(
                    "Predicting mode stats is not implemented yet."
                )
                # (
                #     val_loss,
                #     r_squared,
                #     spearman_corr,
                #     pearson_corr,
                #     val_loss_node_predictions,
                #     val_loss_mode_stats,
                # ) = validate_model_during_training_eign(
                #     config=config,
                #     model=self,
                #     dataset=valid_dl,
                #     loss_func=loss_fct,
                #     device=device,
                #     scalers_validation=scalers_validation,
                # )
                # wandb.log(
                #     {
                #         "val_loss": val_loss,
                #         "epoch": epoch,
                #         "lr": lr,
                #         "r^2": r_squared,
                #         "spearman": spearman_corr,
                #         "pearson": pearson_corr,
                #         "val_loss_node_predictions": val_loss_node_predictions,
                #         "val_loss_mode_stats": val_loss_mode_stats,
                #     }
                # )
            else:
                val_loss, r_squared, spearman_corr, pearson_corr = (
                    validate_model_during_training_eign(
                        config=config,
                        model=self,
                        dataset=valid_dl,
                        loss_func=loss_fct,
                        device=device,
                        scalers_validation=scalers_validation,
                    )
                )
                wandb.log(
                    {
                        "val_loss": val_loss,
                        "epoch": epoch,
                        "lr": lr,
                        "r^2": r_squared,
                        "spearman": spearman_corr,
                        "pearson": pearson_corr,
                    }
                )

            print(
                f"epoch: {epoch}, validation loss: {val_loss}, lr: {lr}, r^2: {r_squared}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if model_save_path:
                    torch.save(self.state_dict(), model_save_path)
                    print(
                        f"Best model saved to {model_save_path} with validation loss: {val_loss}"
                    )

            # Save checkpoint
            if epoch % 20 == 0:
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved to {checkpoint_path}")

            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered. Stopping training.")
                break

        print("Best validation loss: ", best_val_loss)
        wandb.summary["best_val_loss"] = best_val_loss
        wandb.finish()
        return val_loss, epoch    

    def define_layers(self):
        pass

    def initialize_weights(self):
        pass


class EIGNLaplacianConv(EIGN):
    def initialize_block(
        self,
        in_channels_signed: int,
        out_channels_signed: int,
        in_channels_unsigned: int,
        out_channels_unsigned: int,
        signed_activation_fn=F.tanh,
        unsigned_activation_fn=F.relu,
        *args,
        **kwargs,
    ) -> EIGNBlock:
        return EIGNBlockMagneticEdgeLaplacianConv(
            in_channels_signed=in_channels_signed,
            out_channels_signed=out_channels_signed,
            in_channels_unsigned=in_channels_unsigned,
            out_channels_unsigned=out_channels_unsigned,
            signed_activation_fn=signed_activation_fn,
            unsigned_activation_fn=unsigned_activation_fn,
            **kwargs,
        )
