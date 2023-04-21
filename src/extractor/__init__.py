from src.utils.registry import Registry

EXTRACTOR_REGISTRY = Registry("EXTRACTOR")

from src.extractor.bert_extractor import LangExtractor
from src.extractor.base_pc_extractor import PointNetExtractor
from src.extractor.curve_net import CurveNet

EXTRACTOR_REGISTRY.register(PointNetExtractor)
EXTRACTOR_REGISTRY.register(LangExtractor)
EXTRACTOR_REGISTRY.register(CurveNet)
