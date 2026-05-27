# Copyright (c) QIU Tian. All rights reserved.

__all__ = ['Counters']

import torch
from torch import nn


class Counters(nn.Module):
    def __init__(self, num_classes: int, num_features: int):
        super().__init__()

        self.register_buffer('activation_counts', torch.zeros([num_classes, num_features], dtype=torch.int64), persistent=False)
        self.register_buffer('inhibition_counts', torch.zeros([num_classes, num_features], dtype=torch.int64), persistent=False)
        self.register_buffer('total_counts', torch.ones([num_classes], dtype=torch.int64), persistent=False)  # Avoid division by zero

        self.num_classes = num_classes
        self.num_features = num_features

    def update(self, x, targets):
        """
        Args:
            x: torch.Tensor([B, num_features])
            targets: torch.LongTensor([B])
        """
        self.activation_counts.scatter_add_(dim=0,
                                            index=targets[:, None].expand([-1, self.num_features]),
                                            src=torch.where(x > 0, 1, 0))
        self.inhibition_counts.scatter_add_(dim=0,
                                            index=targets[:, None].expand([-1, self.num_features]),
                                            src=torch.where(x == 0, 1, 0))
        self.total_counts += torch.bincount(targets, minlength=self.num_classes)

    def clear(self):
        self.activation_counts.fill_(0)
        self.inhibition_counts.fill_(0)
        self.total_counts.fill_(1)  # Avoid division by zero

    @property
    def activity_levels(self):
        relative_firing_rates = (self.activation_counts - self.inhibition_counts) / self.total_counts[:, None]
        return relative_firing_rates
