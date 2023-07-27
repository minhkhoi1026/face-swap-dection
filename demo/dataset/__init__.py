from src.utils.registry import Registry
DATASET_REGISTRY = Registry("DATASET")

from demo.dataset.face_dataset import FaceSpoofingDataset
from demo.dataset.face_variant_dataset import FaceVariantSpoofingDataset