# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Swin transformer for image segmentation."""
from composer.models.swin_segmentation.model import composer_swin_segmentation as composer_swin_segmentation
from composer.models.swin_segmentation.swin_segmentation_hparams import \
    SwinSegmentationHparams as SwinSegmentationHparams

__all__ = ['composer_swin_segmentation', 'SwinSegmentationHparams']
