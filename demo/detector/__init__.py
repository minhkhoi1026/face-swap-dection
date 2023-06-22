from src.utils.registry import Registry

DETECTOR_REGISTRY = Registry("DETECTOR")

from demo.detector.roop_detector import RoopDetector

DETECTOR_REGISTRY.register(RoopDetector)
