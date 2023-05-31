from src.utils.registry import Registry
DATASET_REGISTRY = Registry("DATASET")

from src.dataset.face_dataset import FaceSpoofingDataset
from src.dataset.face_variant_dataset import FaceVariantSpoofingDataset
