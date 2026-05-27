# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['AIEC']

import torch
import torch.nn.functional as F
from torch import nn

from .counters import Counters
from .tau import TAU


class AIEC(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_classes: int = 1000,
                 tau: float = 0,
                 update_interval: int = 5,
                 global_pool: str = 'token'):
        super().__init__()

        self.tau = TAU(tau=tau)
        self.counters = Counters(num_classes, num_features)

        self.update_interval = update_interval
        self.global_pool = global_pool

        self.decision_weights = nn.Parameter(torch.empty(num_classes, num_features))
        nn.init.xavier_uniform_(self.decision_weights)

        self.count = 0

    def train(self, mode=True):
        if mode is False:
            self.counters.clear()
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def forward(self, x, targets=None, inter_logits=None, inter_irrel_values=None):
        self.targets = targets

        if self.training:
            self.count += 1

        x = self.tau(x)

        if targets is not None:
            if self.global_pool == 'token':
                values = x[:, 0]
            elif self.global_pool == 'avg_token':
                values = x.mean(1)
            elif self.global_pool == 'avg_hw':
                values = x.mean((-1, -2))
            elif self.global_pool == 'none':
                values = x
            else:
                raise ValueError

            logits = F.linear(values, self.decision_weights)
            inter_logits.append(logits)

            if self.training and self.count % self.update_interval == 0:
                predicts = torch.max(logits.detach(), dim=1)[1]
                corrects = predicts == targets
                self.counters.update(values.detach()[corrects], targets[corrects])

            meta_masks = (self.counters.activity_levels <= 0) & (self.decision_weights <= 0)
            masks = meta_masks[targets]
            inter_irrel_values.append(values[masks])

        return x
