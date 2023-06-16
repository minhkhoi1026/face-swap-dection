import torch
from . import METRIC_REGISTRY


from torchmetrics.classification import BinaryRecall as BinaryRecallMetric

@METRIC_REGISTRY.register()
class BinaryRecall(BinaryRecallMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "binary_recall"
