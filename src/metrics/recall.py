import torch
from . import METRIC_REGISTRY


from torchmetrics import Recall as RecallMetric

@METRIC_REGISTRY.register()
class Recall(RecallMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "recall"

