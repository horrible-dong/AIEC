# Copyright (c) QIU Tian. All rights reserved.

from .AIEC import GateLoss
from .cross_entropy import CrossEntropy


def build_criterion(args):
    criterion_name = args.criterion.lower()

    if criterion_name == 'ce':
        return CrossEntropy(losses=['labels'], weight_dict={'loss_ce': 1})

    if criterion_name == 'gate':
        return GateLoss(losses=['labels', 'aux', 'gate'],
                        weight_dict={'loss_ce': 1, 'loss_aux': args.aux_loss_coef, 'loss_gate': args.gate_loss_coef})

    raise ValueError(f"Criterion '{criterion_name}' is not found.")
