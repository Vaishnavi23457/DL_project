"""
bml - Balanced Multimodal Learning (OPM / OGM-GE) toolkit.

Quick API:
    from bml import BalancedModulator, LateFusionModel
    from bml.modulation import OPM, OGMGE, discrepancy_ratios, unimodal_scores_from_logits
"""
from .modulation import (BalancedModulator, OGMGE, OPM, discrepancy_ratios, make_z,
                         unimodal_logits_zero_out, unimodal_scores_from_logits)
from .models import (HFTextEncoder, ImageEncoderResNet18, LateFusionModel, SmallCNN, TextTransformerEncoder,
                     VectorMLPEncoder, build_image_encoder, build_text_encoder, build_vector_encoder)
from .metrics import (accuracy, classification_report, confusion_matrix, expected_calibration_error,
                      macro_f1, per_class_accuracy)
from .xai import head_block_contributions, integrated_gradients_image, modality_attribution, token_importance_text

__version__ = "1.1.0"
__all__ = [
    "BalancedModulator", "OGMGE", "OPM", "discrepancy_ratios", "make_z", "unimodal_logits_zero_out",
    "unimodal_scores_from_logits", "HFTextEncoder", "ImageEncoderResNet18", "LateFusionModel", "SmallCNN",
    "TextTransformerEncoder", "build_image_encoder", "build_text_encoder", "VectorMLPEncoder",
    "build_vector_encoder",
    "accuracy", "classification_report", "confusion_matrix", "expected_calibration_error", "macro_f1",
    "per_class_accuracy", "head_block_contributions", "integrated_gradients_image", "modality_attribution",
    "token_importance_text",
]
