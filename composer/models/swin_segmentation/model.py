# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import functools
import textwrap
import warnings
from typing import Sequence

import torch
import torch.distributed as torch_dist
import torch.nn.functional as F
from torchmetrics import MetricCollection

from composer.loss import loss_registry
from composer.metrics import CrossEntropy, MIoU
from composer.models.initializers import Initializer
from composer.models.tasks import ComposerClassifier
from composer.utils import dist

__all__ = ['swin_segmentation', 'composer_swin_segmentation']


class MMSegSegmentationModel(torch.nn.Module):

    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        logits = self.classifier(features)
        logits = F.interpolate(logits,
                               size=input_shape,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=False)
        return logits


def swin_segmentation(num_classes: int,
                      config_name: str = 'swin_large_patch4_window12_384_22k',
                      is_pretrained: bool = True,
                      sync_bn: bool = True,
                      initializers: Sequence[Initializer] = ()):
    try:
        from mmseg.models import SwinTransformer, UPerHead
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Either mmcv or mmsegmentation is not installed. To install mmcv, please run pip install mmcv-full==1.4.4 -f
             https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html where {cu_version} and
             {torch_version} refer to your CUDA and PyTorch versions, respectively. To install mmsegmentation, please
             run pip install mmsegmentation==0.22.0 on command-line.""")) from e

    # Configs can be found at https://github.com/open-mmlab/mmsegmentation/tree/master/configs/swin
    if config == 'swin_large_patch4_window12_384_22k':
        model = make_swin_large_patch4_window12_384_22k(num_classes, is_pretrained)

    world_size = dist.get_world_size()
    if sync_bn and world_size == 1:
        warnings.warn('sync_bn was true, but only one process is present for training. sync_bn will be ignored.')

    # Apply initializers
    if initializers:
        for initializer in initializers:
            initializer_fn = Initializer(initializer).get_initializer()

            # Only apply initialization to classifier head if pre-trained weights are used
            if is_pretrained is False:
                model.apply(initializer_fn)
            else:
                model.classifier.apply(initializer_fn)

    # Convert to sync batchnorm if necessary.
    if sync_bn and world_size > 1:
        local_world_size = dist.get_local_world_size()

        # List of ranks for each node, assumes that each node has the same number of ranks
        num_nodes = world_size // local_world_size
        process_group = None
        if num_nodes > 1:
            ranks_per_node = [
                list(range(node * local_world_size, (node + 1) * local_world_size)) for node in range(num_nodes)
            ]
            process_groups = [torch_dist.new_group(ranks) for ranks in ranks_per_node]
            process_group = process_groups[dist.get_node_rank()]

        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=process_group)

    return model


def composer_swin_segmentation(num_classes: int,
                               config_name: str = 'swin_large_patch4_window12_384_22k',
                               is_pretrained: bool = True,
                               sync_bn: bool = True,
                               ignore_index: int = -1,
                               cross_entropy_weight: float = 1.0,
                               dice_weight: float = 0.0,
                               initializers: Sequence[Initializer] = ()):
    model = swin_segmentation(num_classes, config_name, is_pretrained, sync_bn, initializers)

    train_metrics = MetricCollection(
        [CrossEntropy(ignore_index=ignore_index),
         MIoU(num_classes, ignore_index=ignore_index)])
    val_metrics = MetricCollection(
        [CrossEntropy(ignore_index=ignore_index),
         MIoU(num_classes, ignore_index=ignore_index)])

    ce_loss_fn = functools.partial(soft_cross_entropy, ignore_index=ignore_index)
    dice_loss_fn = DiceLoss(softmax=True, batch=True, ignore_absent_classes=True)

    def _combo_loss(output, target):
        loss = {}
        if cross_entropy_weight:
            ce_loss = ce_loss_fn(output, target) * cross_entropy_weight
            loss['cross_entropy_loss'] = ce_loss
        if dice_weight:
            dice_loss = dice_loss_fn(output, target) * dice_weight
            loss['dice_loss'] = dice_loss
        return loss

    composer_model = ComposerClassifier(module=model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=_combo_loss)
    return composer_model


def make_swin_large_patch4_window12_384_22k(num_classes: int, is_pretrained: bool = True):
    checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'
    swin_backbone = SwinTransformer(
        pretrain_img_size=384,
        in_channels=3,
        embed_dims=192,
        patch_size=4,
        window_size=12,
        mlp_ratio=4,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
    )
    swin_backbone.init_weights()

    upernet_head = UPerHead(
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
    )

    model = MMSegSegmentationModel(swin_backbone, upernet_head)
    return model
