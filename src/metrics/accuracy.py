from typing import List
import torch
from torch import nn
from . import METRIC_REGISTRY
import numpy as np

from torchmetrics import Metric

@METRIC_REGISTRY.register()
class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.name = "accuracy"
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("targets", default=[], dist_reduce_fx=None)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds >= self.threshold
        self.preds.append(preds)
        self.targets.append(targets)

    def compute(self):
        preds = torch.cat(self.preds)
        targets = torch.cat(self.targets)
        
        return (preds == targets).float().mean()
    
    
# @METRIC_REGISTRY.register()
# class Accuracy:
#     """
#     Accuracy Score
#     """

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.threshold = kwargs.get('threshold') or 0.5
#         self.reset()

#     def update(self, preds, targets):
#         """
#         Perform calculation based on prediction and targets
#         """
#         preds = preds >= self.threshold
#         preds = preds.detach().cpu().float()
#         targets = targets.detach().cpu().float()
#         self.preds += preds.numpy().tolist()
#         self.targets += targets.numpy().tolist()

#     def reset(self):
#         self.targets = []
#         self.preds = []

#     def value(self):
#         score = np.mean(np.array(self.targets) == np.array(self.preds))
#         return {f"accuracy": score}
