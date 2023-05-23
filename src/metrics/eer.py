from typing import List
import torch
from torch import nn
from . import METRIC_REGISTRY
import numpy as np
from sklearn.metrics import roc_curve

import torch
from . import METRIC_REGISTRY


from torchmetrics import Metric

@METRIC_REGISTRY.register()
class BinaryEqualErrorRate(Metric):
    """
    Equal Error Rate for Binary Classification
    """
    def __init__(self):
        super().__init__()
        self.name = "eer"
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds).detach().cpu().numpy()
        targets = torch.cat(self.targets).detach().cpu().numpy()
        # advoid all labels are 0 or 1
        if (targets == 0).all() or (targets == 1).all():
            return 2
        fpr, tpr, threshold = roc_curve(targets, preds, pos_label=1)
        fnr = 1 - tpr
        id = np.nanargmin(np.absolute((fnr - fpr)))
        
        return fpr[id]
