import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
from cassle.distillers.base import base_distill_wrapper
from cassle.losses.vicreg import invariance_loss


def predictive_mse_distill_wrapper(Method=object):
    class PredictiveMSEDistillWrapper(base_distill_wrapper(Method)):
        def __init__(self, distill_lamb: float, distill_proj_hidden_dim, **kwargs):
            super().__init__(**kwargs)

            self.distill_lamb = distill_lamb
            output_dim = kwargs["output_dim"]

            self.distill_predictor = nn.Sequential(
                nn.Linear(output_dim, distill_proj_hidden_dim),
                nn.BatchNorm1d(distill_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(distill_proj_hidden_dim, output_dim),
            )

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("contrastive_distiller")

            parser.add_argument("--distill_lamb", type=float, default=25)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": self.distill_predictor.parameters()},
            ]
            return super().learnable_params + extra_learnable_params

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            z1, z2 = out["z"]
            frozen_z1, frozen_z2 = out["frozen_z"]

            p1 = self.distill_predictor(z1)
            p2 = self.distill_predictor(z2)

            distill_loss = (invariance_loss(p1, frozen_z1) + invariance_loss(p2, frozen_z2)) / 2

            self.log("train_predictive_distill_loss", distill_loss, on_epoch=True, sync_dist=True)

            return out["loss"] + self.distill_lamb * distill_loss

    return PredictiveMSEDistillWrapper
