"""Augmentations realistas para campo (Albumentations): iluminação, poeira/neblina, oclusão, variação de solo, desfoque de movimento.

Cada transformação tem justificativa agronômica/óptica em docs/pt-BR. Caixas são recortadas e descartadas se ficarem
com < 20% da área original (`min_visibility`) — oclusão parcial é esperada, caixa de objeto invisível não.
"""
from __future__ import annotations

import albumentations as A


def build(p: float = 1.0, size: int | None = None) -> A.Compose:
    t = [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),       # sol forte / nublado
        A.RandomGamma(gamma_limit=(70, 140), p=0.3),                                       # exposição automática da câmera
        A.OneOf([A.MotionBlur(blur_limit=(3, 9)), A.GaussianBlur(blur_limit=(3, 7))], p=0.3),  # vibração / foco
        A.RandomFog(fog_coef_range=(0.05, 0.2), alpha_coef=0.1, p=0.2),       # poeira em suspensão (aproximação)
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=25, val_shift_limit=20, p=0.4),  # solo úmido/seco, senescência
        A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(20, 80), hole_width_range=(20, 80), fill=0, p=0.3),  # oclusão por palha/folhas
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),                                       # sensor em pouca luz
    ]
    if size:
        t.append(A.LongestMaxSize(max_size=size))
    return A.Compose(t, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.2), p=p)


def apply(aug: A.Compose, image, boxes, labels):
    """image HxWx3 uint8; boxes lista xyxy em pixels. Devolve (image, boxes, labels) após a transformação."""
    out = aug(image=image, bboxes=boxes, labels=labels)
    return out["image"], [tuple(b) for b in out["bboxes"]], list(out["labels"])
