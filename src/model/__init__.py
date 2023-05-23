from src.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

from src.model.single_head import SingleHeadFrameClassifier
from src.model.double_head import DoubleHeadFrameClassifier

MODEL_REGISTRY.register(SingleHeadFrameClassifier)
MODEL_REGISTRY.register(DoubleHeadFrameClassifier)
