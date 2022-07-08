# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from composer.models.model_hparams import ModelHparams

__all__ = ['SwinSegmentationHparams']


@dataclass
class SwinSegmentationHparams(ModelHparams):

    def validate(self):
        pass

    def initialize_object(self):
        from composer.models.swin_segmentation.model import composer_swin_segmentation

        return composer_swin_segmentation()
