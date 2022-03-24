import argparse
from typing import Any, List, Sequence

import torch
from torch import nn
import torch.nn.functional as F
from cassle.distillers.base import base_distill_wrapper


def cross_entropy(preds, targets):
    return -torch.mean(
        torch.sum(F.softmax(targets, dim=-1) * torch.log_softmax(preds, dim=-1), dim=-1)
    )


def knowledge_distill_wrapper(Method=object):
    class KnowledgeDistillWrapper(base_distill_wrapper(Method)):
        def __init__(
            self,
            distill_lamb: float,
            distill_proj_hidden_dim: int,
            distill_temperature: float,
            **kwargs
        ):
            super().__init__(**kwargs)

            self.distill_lamb = distill_lamb
            self.distill_temperature = distill_temperature
            output_dim = kwargs["output_dim"]
            num_prototypes = kwargs["num_prototypes"]

            self.frozen_prototypes = nn.utils.weight_norm(
                nn.Linear(output_dim, num_prototypes, bias=False)
            )
            for frozen_pg, pg in zip(
                self.frozen_prototypes.parameters(), self.prototypes.parameters()
            ):
                frozen_pg.data.copy_(pg.data)
                frozen_pg.requires_grad = False

            self.distill_predictor = nn.Sequential(
                nn.Linear(output_dim, distill_proj_hidden_dim),
                nn.BatchNorm1d(distill_proj_hidden_dim),
                nn.ReLU(),
                nn.Linear(distill_proj_hidden_dim, output_dim),
            )

            self.distill_prototypes = nn.utils.weight_norm(
                nn.Linear(output_dim, num_prototypes, bias=False)
            )

        @staticmethod
        def add_model_specific_args(
            parent_parser: argparse.ArgumentParser,
        ) -> argparse.ArgumentParser:
            parser = parent_parser.add_argument_group("knowledge_distiller")

            parser.add_argument("--distill_lamb", type=float, default=1)
            parser.add_argument("--distill_proj_hidden_dim", type=int, default=2048)
            parser.add_argument("--distill_temperature", type=float, default=0.1)

            return parent_parser

        @property
        def learnable_params(self) -> List[dict]:
            """Adds distill predictor parameters to the parent's learnable parameters.

            Returns:
                List[dict]: list of learnable parameters.
            """

            extra_learnable_params = [
                {"params": self.distill_predictor.parameters()},
                {"params": self.distill_prototypes.parameters()},
            ]
            return super().learnable_params + extra_learnable_params

        def on_train_start(self):
            super().on_train_start()

            if self.current_task_idx > 0:
                for frozen_pg, pg in zip(
                    self.frozen_prototypes.parameters(), self.prototypes.parameters()
                ):
                    frozen_pg.data.copy_(pg.data)
                    frozen_pg.requires_grad = False

        def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
            out = super().training_step(batch, batch_idx)
            z1, z2 = out["z"]
            frozen_z1, frozen_z2 = out["frozen_z"]

            with torch.no_grad():
                frozen_z1 = F.normalize(frozen_z1)
                frozen_z2 = F.normalize(frozen_z2)
                frozen_p1 = self.frozen_prototypes(frozen_z1) / self.distill_temperature
                frozen_p2 = self.frozen_prototypes(frozen_z2) / self.distill_temperature

            distill_z1 = F.normalize(self.distill_predictor(z1))
            distill_z2 = F.normalize(self.distill_predictor(z2))
            distill_p1 = self.distill_prototypes(distill_z1) / self.distill_temperature
            distill_p2 = self.distill_prototypes(distill_z2) / self.distill_temperature

            distill_loss = (
                cross_entropy(distill_p1, frozen_p1) + cross_entropy(distill_p2, frozen_p2)
            ) / 2

            self.log("train_knowledge_distill_loss", distill_loss, on_epoch=True, sync_dist=True)

            return out["loss"] + self.distill_lamb * distill_loss

    return KnowledgeDistillWrapper
