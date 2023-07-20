from src.utils.registry import Registry
from src.utils.opt import Config

DETECTOR_REGISTRY = Registry("DETECTOR")

from demo.detector.resnet_detector import ResNetDetector
from demo.detector.vit_detector import ViTDetector

DETECTOR_REGISTRY.register(ResNetDetector)
DETECTOR_REGISTRY.register(ViTDetector)

class DetectorFactory:
    def __init__(self, cfg_path="configs/demo/detector_registry_config.yml"):
        self.detector_instances = None
        self.load_config(cfg_path)
    
    def load_config(self, cfg_path):
        self.cfg = Config(cfg_path)
        self.detector_instances = {}
        for detector_config in self.cfg["detectors"]:
            detector_name = detector_config["args"]["name"]
            detector_class = detector_config["class"]
            args = detector_config["args"]
            self.detector_instances[detector_name] = DETECTOR_REGISTRY.get(detector_class)(**args)
            
    def list_all(self):
        return self.detector_instances.keys()
    
    def get(self, name):
        try:
            return self.detector_instances[name]
        except KeyError:
            raise Exception(f"Detector {name} not found! List of available detectors: {self.list_detector()}")

DETECTOR_FACTORY = DetectorFactory()
