# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Progressive Resizing classes and functions."""

from __future__ import annotations

import logging
from functools import partial
from typing import Optional, Tuple, Callable

import torch
import torch.nn.functional as F
from torchvision import transforms

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Tensor
from composer.models.loss import check_for_index_targets

log = logging.getLogger(__name__)

_VALID_MODES = ("crop", "resize")

__all__ = ["resize_batch", "ProgressiveResizing"]


def resize_batch(X: torch.Tensor,
                 y: torch.Tensor,
                 scale_factor: float,
                 mode: str = "resize",
                 resize_targets: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resize inputs and optionally outputs by cropping or interpolating.

    Example:
         .. testcode::

            from composer.algorithms.progressive_resizing import resize_batch
            X_resized, y_resized = resize_batch(
                                        X=X_example,
                                        y=y_example,
                                        scale_factor=0.5,
                                        mode='resize',
                                        resize_targets=False
            )

    Args:
        X: input tensor of shape (N, C, H, W). Resizing will be done along
            dimensions H and W using the constant factor ``scale_factor``.
        y: output tensor of shape (N, H, W) or (N, C, H, W) that will also be resized if
            ``resize_targets`` is ``True``,
        scale_factor: scaling coefficient for the height and width of the
            input/output tensor. 1.0 keeps the original size.
        mode: type of scaling to perform. Value must be one of ``'crop'`` or
            ``'resize'``. ``'crop'`` performs a random crop, whereas ``'resize'``
            performs a nearest neighbor interpolation.
        resize_targets: whether to resize the targets, ``y``, as well

    Returns:
        X_sized: resized input tensor of shape ``(N, C, H * scale_factor, W * scale_factor)``.
        y_sized: if ``resized_targets`` is ``True``, resized output tensor
            of shape ``(N, H * scale_factor, W * scale_factor)`` or  ``(N, C, H * scale_factor, W * scale_factor)``.
            Depending on the input ``y``. Otherwise returns original ``y``.
    """
    # Short-circuit if nothing should be done
    if scale_factor >= 1:
        return X, y

    if mode.lower() == "crop":
        resize_transform = _make_crop(tensor=X, scale_factor=scale_factor)
    elif mode.lower() == "resize":
        resize_transform = _make_resize(scale_factor=scale_factor)
    else:
        raise ValueError(f"Progressive mode '{mode}' not supported.")
        return X, y

    X_sized = resize_transform(X)
    if resize_targets:
        if check_for_index_targets(y):
            # Add a dimension to match shape of the input and change type for resizing
            y_sized = y.float().unsqueeze(1)
            y_sized = resize_transform(y_sized)
            # Convert back to original format for training
            y_sized = y_sized.squeeze(dim=1).to(y.dtype)
        else:
            y_sized = resize_transform(y)
    else:
        y_sized = y
    return X_sized, y_sized


class ProgressiveResizing(Algorithm):
    """Apply Fastai's `progressive resizing <https://\\
    github.com/fastai/fastbook/blob/780b76bef3127ce5b64f8230fce60e915a7e0735/07_sizing_and_tta.ipynb>`__ data
    augmentation to speed up training.

    Progressive resizing initially reduces input resolution to speed up early training.
    Throughout training, the downsampling factor is gradually increased, yielding larger inputs
    up to the original input size. A final finetuning period is then run to finetune the
    model using the full-sized inputs.

    Example:
         .. testcode::

            from composer.algorithms import ProgressiveResizing
            from composer.trainer import Trainer
            progressive_resizing_algorithm = ProgressiveResizing(
                                                mode='resize',
                                                initial_scale=1.0,
                                                finetune_fraction=0.2,
                                                resize_targets=False
                                            )
            trainer = Trainer(
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=eval_dataloader,
                max_duration="1ep",
                algorithms=[progressive_resizing_algorithm],
                optimizers=[optimizer]
            )

    Args:
        mode: Type of scaling to perform. Value must be one of ``'crop'`` or ``'resize'``.
            ``'crop'`` performs a random crop, whereas ``'resize'`` performs a bilinear
            interpolation.
        initial_scale: Initial scale factor used to shrink the inputs. Must be a
            value in between 0 and 1.
        finetune_fraction: Fraction of training to reserve for finetuning on the
            full-sized inputs. Must be a value in between 0 and 1.
        resize_targets: If True, resize targets also.
    """

    def __init__(self,
                 mode: str = 'resize',
                 initial_scale: float = .5,
                 finetune_fraction: float = .2,
                 resize_targets: bool = False):

        if mode not in _VALID_MODES:
            raise ValueError(f"mode '{mode}' is not supported. Must be one of {_VALID_MODES}")

        if not (0 <= initial_scale <= 1):
            raise ValueError(f"initial_scale must be between 0 and 1: {initial_scale}")

        if not (0 <= finetune_fraction <= 1):
            raise ValueError(f"finetune_fraction must be between 0 and 1: {finetune_fraction}")

        self.mode = mode
        self.initial_scale = initial_scale
        self.finetune_fraction = finetune_fraction
        self.resize_targets = resize_targets

    def match(self, event: Event, state: State) -> bool:
        """Run on Event.AFTER_DATALOADER.

        Args:
            event (:class:`Event`): The current event.
            state (:class:`State`): The current state.
        Returns:
            bool: True if this algorithm should run now
        """
        return event == Event.AFTER_DATALOADER

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        """Applies ProgressiveResizing on input images.

        Args:
            event (Event): the current event
            state (State): the current trainer state
            logger (Logger): the training logger
        """
        input, target = state.batch_pair
        assert isinstance(input, Tensor) and isinstance(target, Tensor), \
            "Multiple tensors not supported for this method yet."

        # Calculate the current size of the inputs to use
        initial_size = self.initial_scale
        finetune_fraction = self.finetune_fraction
        scale_frac_elapsed = min([state.get_elapsed_duration().value / (1 - finetune_fraction), 1])

        # Linearly increase to full size at the start of the fine tuning period
        scale_factor = initial_size + (1 - initial_size) * scale_frac_elapsed

        new_input, new_target = resize_batch(X=input,
                                             y=target,
                                             scale_factor=scale_factor,
                                             mode=self.mode,
                                             resize_targets=self.resize_targets)
        state.batch = (new_input, new_target)


def _make_crop(tensor: torch.Tensor, scale_factor: float) -> Callable[[torch.Tensor], torch.Tensor]:
    Hc = int(scale_factor * tensor.shape[2])
    Wc = int(scale_factor * tensor.shape[3])
    top = torch.randint(tensor.shape[2] - Hc, size=(1,))
    left = torch.randint(tensor.shape[3] - Wc, size=(1,))
    resize_transform = partial(transforms.functional.crop, top=top, left=left, height=Hc, width=Wc)
    return resize_transform

def _make_resize(scale_factor: int) -> Callable[[torch.Tensor], torch.Tensor]:
    resize_transform = partial(F.interpolate, scale_factor=scale_factor, mode='nearest')
    return resize_transform