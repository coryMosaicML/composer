# Copyright 2021 MosaicML. All Rights Reserved.

"""Core Exponential Moving Average classes and functions."""

import copy
import logging
from typing import List, Optional, Tuple

import torch

from composer.core import Algorithm, Event, Logger, State
from composer.core.types import Model, Optimizers

log = logging.getLogger(__name__)

__all__ = ["ExponentialMovingAverage", "exponential_moving_average"]


def exponential_moving_average(model: Model, ema_model: Model, alpha: float = 0.9):
    msd = model.state_dict()
    for k, ema_v in ema_model.state_dict().items():
        model_v = msd[k].detach()
        ema_v.copy_(ema_v * alpha + (1. - alpha) * model_v)


class ExponentialMovingAverage(Algorithm):
    """
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self.ema_model = None
        self.training_model = None

    def match(self, event: Event, state: State) -> bool:
        return event in [Event.FIT_START, Event.BATCH_END, Event.EVAL_START, Event.EVAL_END]

    def apply(self, event: Event, state: State, logger: Logger) -> None:
        if event == Event.FIT_START:
            # Initialize the ema model
            self.ema_model = copy.deepcopy(state.model)
            self.training_model = copy.deepcopy(state.model)
        if event == Event.BATCH_END:
            # Update the ema model
            exponential_moving_average(state.model, self.ema_model, alpha=self.alpha)
        if event == Event.EVAL_START:
            # Swap out the training model for the ema model in state
            self.training_model.load_state_dict(state.model.state_dict())
            state.model.load_state_dict(self.ema_model.state_dict())
        if event == Event.EVAL_END:
            # Swap out the ema model for the training model in state
            state.model.load_state_dict(self.training_model.state_dict())
