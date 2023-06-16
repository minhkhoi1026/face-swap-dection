from src.utils.registry import Registry
from torchmetrics import Precision, Recall

METRIC_REGISTRY = Registry("METRIC")

from .accuracy import Accuracy
from .eer import BinaryEqualErrorRate
