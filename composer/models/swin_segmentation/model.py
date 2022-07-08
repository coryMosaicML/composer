# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

import functools
import textwrap

import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection

from composer.loss import loss_registry
from composer.metrics import CrossEntropy, MIoU
from composer.models.tasks import ComposerClassifier

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


def swin_segmentation():
    try:
        from mmseg.models import SwinTransformer, UPerHead
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Either mmcv or mmsegmentation is not installed. To install mmcv, please run pip install mmcv-full==1.4.4 -f
             https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html where {cu_version} and
             {torch_version} refer to your CUDA and PyTorch versions, respectively. To install mmsegmentation, please
             run pip install mmsegmentation==0.22.0 on command-line.""")) from e

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

    upernet_head = UPerHead(
        in_channels=[192, 384, 768, 1536],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=dict(type='SyncBN', requires_grad=True),  #SyncBN on GPU
        align_corners=False,
    )
    model = MMSegSegmentationModel(swin_backbone, upernet_head)
    return model


def composer_swin_segmentation():
    model = swin_segmentation()

    train_metrics = MetricCollection([CrossEntropy(ignore_index=-1), MIoU(150, ignore_index=-1)])
    val_metrics = MetricCollection([CrossEntropy(ignore_index=-1), MIoU(150, ignore_index=-1)])

    loss_fn = loss_registry['soft_cross_entropy']
    loss_fn = functools.partial(loss_fn, ignore_index=-1)

    composer_model = ComposerClassifier(module=model,
                                        train_metrics=train_metrics,
                                        val_metrics=val_metrics,
                                        loss_fn=loss_fn)
    return composer_model
