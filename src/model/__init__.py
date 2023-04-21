from src.utils.registry import Registry

MODEL_REGISTRY = Registry("MODEL")

from src.model.baseline import BaselineModel
from src.model.remove_pc_MLP import RemovePCMLPModel
from src.model.curve_net_bert import CurveNetBertModel
from src.model.unfreeze_bert import UnfreezeBertModel

MODEL_REGISTRY.register(BaselineModel)
MODEL_REGISTRY.register(RemovePCMLPModel)
MODEL_REGISTRY.register(CurveNetBertModel)
MODEL_REGISTRY.register(UnfreezeBertModel)
