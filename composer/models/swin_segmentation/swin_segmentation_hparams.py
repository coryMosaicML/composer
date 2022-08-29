# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import yahp as hp

from composer.models.model_hparams import ModelHparams

__all__ = ['SwinSegmentationHparams']

CONFIG_NAMES = ["swin_large_patch4_window12_384_22k"]


@dataclass
class SwinSegmentationHparams(ModelHparams):
    """`YAHP <https://docs.mosaicml.com/projects/yahp/en/stable/README.html>`_ interface for
        :func:`.swin_segmentation`.

    Args:
        pass
    """

    config_name: str = hp.optional("The name of the mmsegmentation config to use.",
                                   default='swin_large_patch4_window12_384_22k')
    is_pretrained: bool = hp.optional('If true (default), uses the pretrained swin weights', default=True)
    sync_bn: bool = hp.optional('If true, use SyncBatchNorm to sync batch norm statistics across GPUs.', default=True)
    ignore_index: int = hp.optional('Class label to ignore when calculating the loss and other metrics.', default=-1)
    cross_entropy_weight: float = hp.optional('Weight to scale the cross entropy loss.', default=1.0)
    dice_weight: float = hp.optional('Weight to scale the dice loss.', default=0.0)

    def validate(self):
        if self.num_classes is None:
            raise ValueError('num_classes must be specified')

        if self.cross_entropy_weight < 0:
            raise ValueError(f'cross_entropy_weight value {self.cross_entropy_weight} must be positive or zero.')

        if self.dice_weight < 0:
            raise ValueError(f'dice_weight value {self.dice_weight} must be positive or zero.')

        if self.cross_entropy_weight == 0 and self.dice_weight == 0:
            raise ValueError('Both cross_entropy_weight and dice_weight cannot be zero.')

    def initialize_object(self):
        from composer.models.swin_segmentation.model import composer_swin_segmentation

        if self.num_classes is None:
            raise ValueError('num_classes must be specified')

        return composer_swin_segmentation(num_classes=self.num_classes)
