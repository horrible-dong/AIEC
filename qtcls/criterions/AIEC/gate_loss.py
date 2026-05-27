# Copyright (c) QIU Tian. All rights reserved.

import torch
import torch.nn.functional as F

from ..cross_entropy import CrossEntropy

__all__ = ['GateLoss']


class GateLoss(CrossEntropy):
    def __init__(self, losses: list, weight_dict: dict):
        super().__init__(losses, weight_dict)

    def loss_aux(self, outputs, targets, **kwargs):
        inter_logits = outputs['inter_logits']
        num_layers, batch_size, num_classes = inter_logits.shape
        loss_aux = F.cross_entropy(inter_logits.view(-1, num_classes), targets.repeat(num_layers), reduction='mean')
        losses = {'loss_aux': loss_aux}
        return losses

    def loss_gate(self, outputs, targets, **kwargs):
        values = outputs['inter_irrel_values']
        loss_gate = (values ** 2).mean() if len(values) > 0 else torch.tensor(0.).to(values.device)
        losses = {'loss_gate': loss_gate}
        return losses
