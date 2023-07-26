from src.utils.registry import Registry
DATASET_REGISTRY = Registry("DATASET")

from demo.dataset.fafi_dataset import FafiDataset
from demo.dataset.msr_dataset import MsrDataset