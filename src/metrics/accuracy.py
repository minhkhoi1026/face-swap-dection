import torch
from . import METRIC_REGISTRY


from torchmetrics import Metric

@METRIC_REGISTRY.register()
class Accuracy(Metric):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.name = "accuracy"
        self.threshold = threshold
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
