from src.utils.registry import Registry

METRIC_REGISTRY = Registry("METRIC")

from .accuracy import Accuracy
from .eer import BinaryEqualErrorRate
from .precision import BinaryPrecision
from .recall import BinaryRecall
