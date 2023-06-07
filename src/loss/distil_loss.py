import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Softmax
from src.model import MODEL_REGISTRY
from src.utils.opt import parse_config

class DistillationLoss(nn.Module):
    """
    Distillation Loss for Knowledge Distillation
    Need to initial with teacher model.
    Receive logits from student model and compute loss with soft label from teacher model.
    """
    def __init__(self, 
                 teacher_config_file: str,
                 config_type: str, 
                 weight_path: str,
                 temperature: float = 1.0,
                 reduction='mean'):
        """
        Args:
            teacher_config_file (str): path to teacher model configuration dictionary
            config_type (str): type of teacher config file, could either be "raw" or "wandb"
            weight_path (str): path to teacher model weight
            temperature (float): temperature for soft label generation
            reduction (str): reduction method for loss
        """
        super().__init__()
        teacher_model_config = parse_config(teacher_config_file, mode=config_type)
        teacher_model = MODEL_REGISTRY.get(teacher_model_config["model"]["name"])(teacher_model_config)
        self.teacher_model = teacher_model.load_from_checkpoint(weight_path,
                                       cfg=teacher_model_config,
                                       strict=True)
        self.teacher_model.eval()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, student_logits, imgs):
        with torch.no_grad():
            # TODO: create general structure dependency of teacher model
            teacher_logits = self.teacher_model({"imgs": imgs})["logits"] / self.temperature
            teacher_prob = teacher_logits.softmax(dim=1)
        return CrossEntropyLoss(student_logits, teacher_prob, self.reduction)
