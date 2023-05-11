from src.utils.registry import Registry

METRIC_REGISTRY = Registry("METRIC")

from .accuracy import Accuracy
from .eer import BinaryEqualErrorRate
