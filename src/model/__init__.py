from src.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

from src.model.single_head import SingleHeadFrameClassifier
from src.model.double_head import DoubleHeadFrameClassifier
from src.model.distillation import DistillationFrameClassifier
from src.model.double_head_attention import DoubleHeadAttentionFrameClassifier

MODEL_REGISTRY.register(SingleHeadFrameClassifier)
MODEL_REGISTRY.register(DoubleHeadFrameClassifier)
MODEL_REGISTRY.register(DistillationFrameClassifier)
MODEL_REGISTRY.register(DoubleHeadAttentionFrameClassifier)
