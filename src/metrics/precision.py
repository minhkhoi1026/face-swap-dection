import torch
from . import METRIC_REGISTRY


from torchmetrics.classification import BinaryPrecision as BinaryPrecisionMetric

@METRIC_REGISTRY.register()
class BinaryPrecision(BinaryPrecisionMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "binary_precision"

if __name__ == "__main__":
    x = BinaryPrecision()
    print(x.name)
